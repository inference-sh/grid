"""Text to speech shared by the per-model ElevenLabs apps (eleven-v4,
eleven-v4-turbo, eleven-v3, eleven-multilingual-v2, eleven-flash-v2-5).

Each app is one model at one price. This module holds what they have in
common: the voices, the inputs, `speak` (a whole text in, an audio file out),
`converse` (several voices in one take, for the v4 and v3 apps) and `relay`
(the live function of the v4 models: text streamed in, speech streamed back
over ElevenLabs' Text to Dialogue WebSocket).

Symlink this file into the app folder next to elevenlabs_helper.py.

What the socket of the live function carries:

    caller -> app   {"events": {"type": "text", "text": "Hello, "}}   text to speak, in pieces
    caller -> app   {"events": {"type": "flush"}}                     speak what was sent without waiting for more
    caller -> app   {"events": {"type": "end"}}                       no more text: finish speaking, then end
    app -> caller   {"characters": 0}                                 the first frame: the app is there
    app -> caller   <binary PCM s16le mono 24 kHz>                    an item of RealtimeOutput.audio
    app -> caller   {"characters": 42, "seconds": 2.7}                a stretch of speech is complete
    caller closes  ->  the session ends at once; `end` lets the speech finish first

ElevenLabs waits for about 40 characters and 8 words before it starts
speaking, so short lines need a `flush`. The voice and its settings are fixed
when the session opens; only ``idle_minutes`` can change mid-stream.
"""

import asyncio
import base64
import json
import os
import tempfile
import time
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional, Union
from urllib.parse import urlencode

import httpx
from inferencesh import AudioMeta, BaseAppInput, BaseAppOutput, File, Live, OutputMeta, PCM16, Socket, Stream
from pydantic import BaseModel, Field

from .elevenlabs_helper import get_audio_duration, get_voice_id, text_to_speech

DIALOGUE_URL = "wss://api.elevenlabs.io/v1/text-to-dialogue/stream-input"
DIALOGUE_REST_URL = "https://api.elevenlabs.io/v1/text-to-dialogue"
MAX_DIALOGUE_VOICES = 10
SAMPLE_RATE = 24000
BYTES_PER_SECOND = SAMPLE_RATE * 2
KEEP_ALIVE_SECONDS = 10.0           # ElevenLabs ends a session it hears nothing from for 20 seconds
FLUSH_SECONDS = 20.0                # how long ElevenLabs gets to finish speaking after `end`

PremadeVoice = Literal[
    "adam",      # American male, dominant/firm
    "alice",     # British female, clear/engaging
    "aria",      # American female, expressive
    "bella",     # American female, professional/warm
    "bill",      # American male, wise/mature
    "brian",     # American male, deep/comforting
    "callum",    # American male, husky
    "charlie",   # Australian male, deep/energetic
    "chris",     # American male, charming
    "daniel",    # British male, broadcaster
    "eric",      # American male, smooth/trustworthy
    "george",    # British male, warm storyteller
    "harry",     # American male, fierce/rough
    "jessica",   # American female, playful/bright
    "laura",     # American female, quirky/sassy
    "liam",      # American male, energetic
    "lily",      # British female, velvety
    "matilda",   # American female, professional
    "river",     # American neutral, calm/informative
    "roger",     # American male, laid-back
    "sarah",     # American female, confident
    "will",      # American male, relaxed
]


OutputFormat = Literal[
    "mp3_44100_128",
    "mp3_44100_192",
    "pcm_16000",
    "pcm_22050",
    "pcm_24000",
    "pcm_44100",
]


class SpeechInput(BaseAppInput):
    """What every model takes. An app overrides `text` to state its own limit."""

    text: str = Field(description="Text to convert to speech.")
    voice: PremadeVoice = Field(
        default="george",
        description="Premade voice to use. Ignored if voice_id is provided.",
    )
    voice_id: Optional[str] = Field(
        default=None,
        description="Custom voice ID (e.g. from elevenlabs/voice-clone). Overrides the voice field when provided.",
    )
    output_format: OutputFormat = Field(
        default="mp3_44100_128",
        description="Audio output format. mp3_44100_128 is standard quality MP3.",
    )
    stability: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Voice stability (0-1). Higher = more consistent, lower = more expressive.",
    )
    similarity_boost: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Similarity boost (0-1). Higher = closer to original voice.",
    )


class StyledSpeechInput(SpeechInput):
    """The models before v4 also take a style and a speaker boost."""

    style: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Style exaggeration (0-1). Increases expressiveness but may reduce stability.",
    )
    use_speaker_boost: bool = Field(
        default=True,
        description="Enable speaker boost for enhanced clarity.",
    )


class SpeechOutput(BaseAppOutput):
    audio: File = Field(description="Generated speech audio file")


def speech_meta(model: str, characters: int, seconds: float, sample_rate: Optional[int] = None) -> OutputMeta:
    """ElevenLabs bills the characters it was sent: the speech is one audio
    output carrying them and the model. Both functions report this shape."""
    extra = {"characters": characters, "model": model}
    audio = AudioMeta(seconds=round(seconds, 3), extra=extra)
    if sample_rate:
        audio.sample_rate = sample_rate
    return OutputMeta(inputs=[], outputs=[audio])


async def speak(app: Any, model: str, max_chars: int, input_data: SpeechInput) -> SpeechOutput:
    """A whole text in, an audio file out."""
    if len(input_data.text) > max_chars:
        raise ValueError(f"Text exceeds the {max_chars} character limit of {model}")

    voice_id = input_data.voice_id or get_voice_id(input_data.voice)
    voice_settings = {"stability": input_data.stability, "similarity_boost": input_data.similarity_boost}
    if isinstance(input_data, StyledSpeechInput):
        voice_settings["style"] = input_data.style
        voice_settings["use_speaker_boost"] = input_data.use_speaker_boost

    app.logger.info("speaking %d characters with %s, voice %s", len(input_data.text), model, voice_id)
    path = await asyncio.to_thread(
        text_to_speech,
        text=input_data.text,
        voice_id=voice_id,
        model_id=model,
        output_format=input_data.output_format,
        voice_settings=voice_settings,
        logger=app.logger,
    )
    seconds = _audio_seconds(app, path, input_data.output_format)
    return SpeechOutput(audio=File(path=path), output_meta=speech_meta(model, len(input_data.text), seconds))


def _audio_seconds(app: Any, path: str, output_format: str) -> float:
    if output_format.startswith("pcm_"):
        return os.path.getsize(path) / (int(output_format.split("_")[1]) * 2)
    return get_audio_duration(path, app.logger)


# ------------------------------------------------------------ dialogue


class DialogueSegment(BaseModel):
    """One speaker's line."""

    text: str = Field(description="What this speaker says. Audio tags such as [laughs], [interrupting] or [whispers] are performed, not read aloud.")
    voice: PremadeVoice = Field(default="george", description="Premade voice for this line. Ignored if voice_id is provided.")
    voice_id: Optional[str] = Field(
        default=None,
        description="Custom voice ID (e.g. from elevenlabs/voice-clone). Overrides the voice field when provided.",
    )


class DialogueInput(BaseAppInput):
    segments: List[DialogueSegment] = Field(
        min_length=1,
        description="The lines of the conversation, in order, each with its own voice. Up to 10 different voices. ElevenLabs recommends at most 2,000 characters in total for a reliable take; split a longer script into several runs.",
    )
    output_format: OutputFormat = Field(
        default="mp3_44100_128",
        description="Audio output format. mp3_44100_128 is standard quality MP3.",
    )
    language_code: Optional[str] = Field(
        default=None, description="Language of the text as an ISO 639-1 code such as en. Left empty it is detected."
    )


async def converse(app: Any, model: str, max_chars: int, input_data: DialogueInput) -> SpeechOutput:
    """Several voices in one take: a list of lines in, one audio file out."""
    inputs = [
        {"text": segment.text, "voice_id": segment.voice_id or get_voice_id(segment.voice)}
        for segment in input_data.segments
    ]
    characters = sum(len(segment.text) for segment in input_data.segments)
    voices = {line["voice_id"] for line in inputs}
    if characters > max_chars:
        raise ValueError(f"The lines add up to {characters} characters; {model} takes {max_chars} in one dialogue")
    if len(voices) > MAX_DIALOGUE_VOICES:
        raise ValueError(f"A dialogue takes up to {MAX_DIALOGUE_VOICES} different voices; this one has {len(voices)}")

    body: Dict[str, Any] = {"inputs": inputs, "model_id": model}
    if input_data.language_code:
        body["language_code"] = input_data.language_code

    app.logger.info("dialogue: %d lines, %d voices, %d characters with %s", len(inputs), len(voices), characters, model)
    async with httpx.AsyncClient(timeout=httpx.Timeout(600, connect=30)) as client:
        response = await client.post(
            DIALOGUE_REST_URL,
            params={"output_format": input_data.output_format},
            headers={"xi-api-key": app.api_key},
            json=body,
        )
    if response.status_code != 200:
        raise RuntimeError(f"ElevenLabs answered {response.status_code}: {response.text[:500]}")

    suffix = ".pcm" if input_data.output_format.startswith("pcm_") else ".mp3"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as audio:
        audio.write(response.content)
    seconds = _audio_seconds(app, audio.name, input_data.output_format)
    return SpeechOutput(audio=File(path=audio.name), output_meta=speech_meta(model, characters, seconds))


# ------------------------------------------------------------ the live function


class Text(BaseModel):
    """Text to speak. It may arrive in pieces; speech starts once enough of it has (about 40 characters and 8 words) or at a flush."""

    type: Literal["text"] = "text"
    text: str = Field(description="What to say. Audio tags such as [laughs] or [whispers] are spoken as directions.")
    new_turn: bool = Field(default=False, description="True when this starts a new turn, so the delivery resets")


class Flush(BaseModel):
    """Speak what has been sent so far without waiting for more."""

    type: Literal["flush"] = "flush"


class End(BaseModel):
    """No more text: finish speaking what was sent, then end the session."""

    type: Literal["end"] = "end"


class RealtimeInput(BaseAppInput):
    events: Stream[Union[Text, Flush, End]] = Field(description="Text to speak as it is written, a flush, or the end of the text")
    voice: PremadeVoice = Field(
        default="george",
        description="Premade voice to use. Ignored if voice_id is provided.",
    )
    voice_id: Optional[str] = Field(
        default=None,
        description="Custom voice ID (e.g. from elevenlabs/voice-clone). Overrides the voice field when provided.",
    )
    language_code: Optional[str] = Field(
        default=None, description="Language of the text as an ISO 639-1 code such as en. Left empty it is detected."
    )
    stability: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Voice stability (0-1). Higher = more consistent, lower = more expressive.",
    )
    idle_minutes: float = Field(
        default=2.0, ge=0, le=60, description="End the session after this long with no text sent (0: never)"
    )


class RealtimeOutput(BaseAppOutput):
    audio: Stream[PCM16(SAMPLE_RATE)] = Field(description="The speech")
    characters: int = Field(default=0, description="Characters sent to be spoken so far")
    seconds: float = Field(default=0, description="Speech generated so far, in seconds")
    partial: bool = Field(default=True, description="True while the session goes on; False on the result")
    end_reason: str = Field(default="", description="Why the session ended, on the result")


_DONE = object()


class _Ended:
    """The session is over though the caller is still there: what was spoken so far is the result."""

    def __init__(self, reason: str):
        self.reason = reason


async def stop_reading(app: Any) -> bool:
    """An app's on_cancel: stop reading from the caller, so the session winds down."""
    socket, app._socket = getattr(app, "_socket", None), None
    if socket is not None:
        await socket.close()
    return True


async def relay(app: Any, model: str, input_data: RealtimeInput, socket: Socket) -> AsyncGenerator[RealtimeOutput, None]:
    """Speak text as it is written: the app's live function."""
    import websockets

    live = Live(socket, input_data, RealtimeOutput)
    app._socket = socket
    voice_id = input_data.voice_id or get_voice_id(input_data.voice)
    flushed = asyncio.Event()               # ElevenLabs has spoken everything it was sent
    snapshots: "asyncio.Queue[Any]" = asyncio.Queue()
    state: Dict[str, Any] = {
        "characters": 0,                    # characters sent to ElevenLabs
        "audio": 0,                         # bytes of speech it sent back
        "text_at": time.monotonic(),        # when the caller last sent text
        "sent_at": time.monotonic(),        # when ElevenLabs was last sent anything
        "ending": False,                    # the caller asked to end, or left
    }
    eleven_error: Dict[str, str] = {}

    query = {"model_id": model, "output_format": f"pcm_{SAMPLE_RATE}"}
    if input_data.language_code:
        query["language_code"] = input_data.language_code
    eleven = await websockets.connect(
        f"{DIALOGUE_URL}?{urlencode(query)}",
        additional_headers={"xi-api-key": app.api_key},
        max_size=None,
    )
    # The first message registers the voice; ElevenLabs does not answer it.
    await eleven.send(json.dumps({"voices": [voice_id], "voice_settings": {"stability": input_data.stability}}))

    async def tell(message: Dict[str, Any]) -> None:
        state["sent_at"] = time.monotonic()
        await eleven.send(json.dumps(message))

    async def progress() -> None:
        if not socket.closed:                 # the result is yielded after the caller has gone
            await live.send(characters=state["characters"], seconds=round(state["audio"] / BYTES_PER_SECOND, 3))

    async def uplink() -> None:
        ended = False
        async for update in live:
            if update.field != "events":
                continue                      # an ordinary field; Live already applied it
            event = update.value
            if isinstance(event, Text):
                if not event.text:
                    continue
                state["characters"] += len(event.text)
                state["text_at"] = time.monotonic()
                await tell({"inputs": [{"text": event.text, "voice_id": voice_id, "new_turn": event.new_turn}]})
            elif isinstance(event, Flush):
                await tell({"flush": True})
            elif isinstance(event, End):
                ended = True
                break
        # `end` lets ElevenLabs finish speaking: the rest of the audio still
        # reaches the caller. A caller that just left gets no more of it.
        state["ending"] = True
        try:
            await tell({"close_socket": True})
            if ended:
                await asyncio.wait_for(flushed.wait(), FLUSH_SECONDS)
        except Exception:  # noqa: BLE001 - ElevenLabs is gone or slow; what was spoken is the result
            pass
        snapshots.put_nowait(_DONE)

    async def keep_alive() -> None:
        while not state["ending"]:
            await asyncio.sleep(1.0)
            if not state["ending"] and time.monotonic() - state["sent_at"] >= KEEP_ALIVE_SECONDS:
                await tell({"keep_alive": True})

    async def downlink() -> None:
        try:
            async for message in eleven:
                if isinstance(message, bytes):
                    continue
                event = json.loads(message)
                if event.get("error"):
                    eleven_error["message"] = f"ElevenLabs: {event.get('message') or event.get('error')}"
                    app.logger.warning("%s (%s)", eleven_error["message"], event.get("error"))
                    await live.error(eleven_error["message"])
                    continue
                if event.get("audio"):
                    pcm = base64.b64decode(event["audio"])
                    state["audio"] += len(pcm)
                    if not socket.closed:
                        await live.send(audio=pcm)
                if event.get("is_final_audio_for_turn"):
                    await progress()
                    snapshots.put_nowait(None)
                if event.get("is_final"):
                    break
        except asyncio.CancelledError:
            raise
        except Exception as err:  # noqa: BLE001 - a closed socket ends the session, it does not fail it
            eleven_error.setdefault("message", f"ElevenLabs closed the session: {err}")
        flushed.set()
        if state["ending"]:
            return                              # the caller ended first; the result is on its way
        # ElevenLabs is gone. A session that never spoke was refused, which
        # is a failure; one that did ends like any other and is billed.
        reason = eleven_error.get("message") or "ElevenLabs closed the session"
        snapshots.put_nowait(_Ended(reason) if state["audio"] else RuntimeError(reason))

    async def idle_watch() -> None:
        while True:
            limit = input_data.idle_minutes * 60      # read each time: it can change mid-stream
            await asyncio.sleep(min(5.0, limit / 4) if limit > 0 else 5.0)
            if limit <= 0 or time.monotonic() - state["text_at"] < limit:
                continue
            minutes = f"{input_data.idle_minutes:g} minute{'' if input_data.idle_minutes == 1 else 's'}"
            reason = f"ended after {minutes} with no text"
            app.logger.info("%s", reason)
            await live.error(reason)
            snapshots.put_nowait(_Ended(reason))
            return

    def snapshot(partial: bool = True) -> RealtimeOutput:
        return RealtimeOutput(
            characters=state["characters"], seconds=round(state["audio"] / BYTES_PER_SECOND, 3), partial=partial
        )

    await progress()                            # the first frame: the app is there
    end_reason = "the caller ended the session"
    tasks = [asyncio.create_task(job()) for job in (uplink, downlink, keep_alive, idle_watch)]
    try:
        while True:
            item = await snapshots.get()
            if item is _DONE:
                break
            if isinstance(item, _Ended):
                end_reason = item.reason
                break
            if isinstance(item, BaseException):
                raise item
            yield snapshot()
    finally:
        app._socket = None
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await eleven.close()

    seconds = state["audio"] / BYTES_PER_SECOND
    result = snapshot(partial=False)
    result.end_reason = end_reason
    result.output_meta = speech_meta(model, state["characters"], seconds, SAMPLE_RATE)
    app.logger.info("session ended (%s): %d characters, %.1fs of speech", end_reason, state["characters"], seconds)
    yield result

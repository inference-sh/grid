"""
ElevenLabs Text to Speech

High-quality text-to-speech using ElevenLabs models.
v4 and v3 are the most expressive, with audio tags for emotion/style control.

`run` takes the whole text and returns an audio file.

`realtime` is a live function: the caller streams text as it is written (an
LLM's tokens, say) and hears it spoken with about a tenth of a second of
latency. The app relays ElevenLabs' Text to Dialogue WebSocket
(``wss://api.elevenlabs.io/v1/text-to-dialogue/stream-input``), the only
realtime endpoint that takes Eleven v4.

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
import logging
import time
from typing import Any, AsyncGenerator, Dict, Literal, Optional, Union
from urllib.parse import urlencode

from inferencesh import (
    AudioMeta,
    BaseApp,
    BaseAppInput,
    BaseAppOutput,
    File,
    Live,
    OutputMeta,
    PCM16,
    Socket,
    Stream,
)
from pydantic import BaseModel, Field

from .elevenlabs_helper import text_to_speech, get_api_key, get_voice_id, get_audio_duration

DIALOGUE_URL = "wss://api.elevenlabs.io/v1/text-to-dialogue/stream-input"
SAMPLE_RATE = 24000
BYTES_PER_SECOND = SAMPLE_RATE * 2
KEEP_ALIVE_SECONDS = 10.0           # ElevenLabs ends a session it hears nothing from for 20 seconds
FLUSH_SECONDS = 20.0                # how long ElevenLabs gets to finish speaking after `end`
# Characters a model takes in one request; the others take 40,000.
MAX_CHARS = {"eleven_v4": 10000, "eleven_v3": 5000}
TAGGED_MODELS = ("eleven_v4", "eleven_v3")


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


class AppInput(BaseAppInput):
    """Input schema for ElevenLabs TTS."""

    text: str = Field(
        description="Text to convert to speech. Max 10,000 characters for v4, 5,000 for v3, 40,000 for v2 models.",
    )
    voice: PremadeVoice = Field(
        default="george",
        description="Premade voice to use. Ignored if voice_id is provided.",
    )
    voice_id: Optional[str] = Field(
        default=None,
        description="Custom voice ID (e.g. from elevenlabs/voice-clone). Overrides the voice field when provided.",
    )
    model: Literal[
        "eleven_v4",
        "eleven_v3",
        "eleven_multilingual_v2",
        "eleven_turbo_v2_5",
        "eleven_flash_v2_5",
    ] = Field(
        default="eleven_v3",
        description="Model to use. v4 is the newest and most expressive, with 90+ languages and free-form audio tags; it takes stability and similarity_boost only. v3 is the previous generation with 70+ languages and audio tags, multilingual_v2 is high quality, turbo/flash are faster with lower latency.",
    )
    audio_tags: bool = Field(
        default=False,
        description="Enable audio tags in text for emotion/style control (v4 and v3 only). Use tags like [laughs], [whispers], [excited], [sad], [slow], [fast], [shouts], [sighs] inline in your text.",
    )
    output_format: Literal[
        "mp3_44100_128",
        "mp3_44100_192",
        "pcm_16000",
        "pcm_22050",
        "pcm_24000",
        "pcm_44100",
    ] = Field(
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


class AppOutput(BaseAppOutput):
    """Output schema for ElevenLabs TTS."""
    audio: File = Field(description="Generated speech audio file")


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
    model: Literal["eleven_v4_turbo", "eleven_v4"] = Field(
        default="eleven_v4_turbo",
        description="eleven_v4_turbo starts speaking in about 100 ms; eleven_v4 is slower and the most expressive.",
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


class App(BaseApp):
    """ElevenLabs TTS app implementation."""

    async def setup(self):
        """Initialize the application."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.api_key = get_api_key()
        self._socket: Optional[Socket] = None
        self.logger.info("ElevenLabs TTS app initialized")

    async def on_cancel(self):
        # Stop reading from the caller: the uplink loop ends and the session winds down.
        socket, self._socket = self._socket, None
        if socket is not None:
            await socket.close()
        return True

    async def run(self, input_data: AppInput) -> AppOutput:
        """Generate speech from text."""
        max_chars = MAX_CHARS.get(input_data.model, 40000)
        if len(input_data.text) > max_chars:
            raise ValueError(f"Text exceeds {max_chars} character limit for {input_data.model}")

        if input_data.audio_tags and input_data.model not in TAGGED_MODELS:
            raise ValueError("Audio tags are only supported with the eleven_v4 and eleven_v3 models")

        resolved_voice_id = input_data.voice_id if input_data.voice_id else get_voice_id(input_data.voice)

        self.logger.info(f"Generating speech: {len(input_data.text)} characters")
        self.logger.info(f"Voice ID: {resolved_voice_id}, Model: {input_data.model}")

        voice_settings = {
            "stability": input_data.stability,
            "similarity_boost": input_data.similarity_boost,
        }
        if input_data.model != "eleven_v4":
            # v4 has no style or speaker boost: stability and similarity are its two settings.
            voice_settings["style"] = input_data.style
            voice_settings["use_speaker_boost"] = input_data.use_speaker_boost

        audio_path = text_to_speech(
            text=input_data.text,
            voice_id=resolved_voice_id,
            model_id=input_data.model,
            output_format=input_data.output_format,
            voice_settings=voice_settings,
            logger=self.logger,
        )

        duration = get_audio_duration(audio_path, self.logger)

        return AppOutput(
            audio=File(path=audio_path),
            output_meta=OutputMeta(
                inputs=[],
                outputs=[AudioMeta(
                    seconds=duration,
                    extra={"characters": len(input_data.text), "model": input_data.model}
                )]
            )
        )

    async def realtime(self, input_data: RealtimeInput, socket: Socket) -> AsyncGenerator[RealtimeOutput, None]:
        """Speak text as it is written."""
        import websockets

        live = Live(socket, input_data, RealtimeOutput)
        self._socket = socket
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

        query = {"model_id": input_data.model, "output_format": f"pcm_{SAMPLE_RATE}"}
        if input_data.language_code:
            query["language_code"] = input_data.language_code
        eleven = await websockets.connect(
            f"{DIALOGUE_URL}?{urlencode(query)}",
            additional_headers={"xi-api-key": self.api_key},
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
                        self.logger.warning("%s (%s)", eleven_error["message"], event.get("error"))
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
                self.logger.info("%s", reason)
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
            self._socket = None
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await eleven.close()

        seconds = state["audio"] / BYTES_PER_SECOND
        result = snapshot(partial=False)
        result.end_reason = end_reason
        # ElevenLabs bills the characters it was sent. Same shape as `run`:
        # the speech as one audio output carrying the characters and the model.
        result.output_meta = OutputMeta(
            inputs=[],
            outputs=[
                AudioMeta(
                    seconds=round(seconds, 3),
                    sample_rate=SAMPLE_RATE,
                    extra={"characters": state["characters"], "model": input_data.model},
                )
            ],
        )
        self.logger.info("session ended (%s): %d characters, %.1fs of speech", end_reason, state["characters"], seconds)
        yield result

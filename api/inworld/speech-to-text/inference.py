"""
Inworld Speech to Text

Multi-provider speech transcription with optional word timestamps
and voice profile identification.

`run` takes an audio file. `realtime` is a live function: the caller streams
microphone audio and reads the transcript as it forms. The app relays
Inworld's streaming endpoint
(``wss://api.inworld.ai/stt/v1/transcribe:streamBidirectional``), which answers
with an interim transcript of the turn in progress and a final one when the
speaker pauses.

    caller -> app   <binary PCM s16le mono 16 kHz>   an item of RealtimeInput.audio (a frame every ~20 ms)
    app -> caller   {"text": ""}                     the first frame: the stream to Inworld is open
    app -> caller   {"text": "what's the wea"}       the transcript so far, its tail still changing
    caller closes  ->  Inworld finishes the turn and the function yields its result

The socket is the low-latency channel; the yields are the task's output. Each
finished turn yields a cumulative snapshot, and the yield after the socket
closes is the result and carries ``output_meta``.

Inworld takes its options in the first message of the stream, so the ordinary
inputs of `realtime` are fixed once it is open; only ``idle_minutes`` can
change mid-stream. While the caller sends nothing the app sends silence, so a
pause is heard as one and ends the turn.
"""

import asyncio
import base64
import json
import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

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

from .inworld_helper import speech_to_text, get_api_key, get_auth_header, get_audio_duration

MODEL = "inworld/inworld-stt-1"
REALTIME_URL = "wss://api.inworld.ai/stt/v1/transcribe:streamBidirectional"
SAMPLE_RATE = 16000
BYTES_PER_SECOND = SAMPLE_RATE * 2
CHUNK_BYTES = BYTES_PER_SECOND // 10    # 100 ms, the size Inworld's own samples send
GAP_SECONDS = 0.5                       # the caller has gone quiet: fill with silence from here on
FLUSH_SECONDS = 3.0                     # how long Inworld gets to finish the turn after the caller closes
SILENCE = bytes(CHUNK_BYTES)


class AppInput(BaseAppInput):
    """Input schema for Inworld STT."""

    audio: File = Field(
        description="Audio file to transcribe (MP3, WAV, FLAC, OGG, PCM).",
    )
    language: Optional[str] = Field(
        default=None,
        description="BCP-47 language code (e.g. 'en-US', 'ja-JP'). Auto-detected if omitted.",
    )
    include_word_timestamps: bool = Field(
        default=False,
        description="Include per-word timing information in the output.",
    )


class AppOutput(BaseAppOutput):
    """Output schema for Inworld STT."""
    text: str = Field(description="Full transcription text")
    words: Optional[List[dict]] = Field(default=None, description="Word-level timestamps with confidence scores")


class RealtimeInput(BaseAppInput):
    audio: Stream[PCM16(SAMPLE_RATE)] = Field(description="Microphone audio, a frame every 20 ms or so")
    language: Optional[str] = Field(
        default=None,
        description="Language hint such as 'en' or 'ja' (a BCP-47 code is reduced to its base). Auto-detected if omitted.",
    )
    keyterms: List[str] = Field(
        default_factory=list,
        description="Names and terms the transcript should prefer. Letters, digits, spaces and basic punctuation only.",
    )
    silence_ms: Optional[int] = Field(
        default=None, ge=20, le=5000, description="Silence that ends a turn at the latest, in ms. Left empty Inworld decides."
    )
    idle_minutes: float = Field(
        default=2.0,
        ge=0,
        le=60,
        description="End the session after this long with nobody speaking (0: never). Inworld bills every minute of audio it is sent, silent or not.",
    )


class Turn(BaseModel):
    """A finished turn. Its times are seconds from the start of the stream, when Inworld reports them."""

    text: str = Field(description="What was said")
    start: float = Field(default=0, description="Seconds from the start of the stream")
    end: float = Field(default=0, description="Seconds from the start of the stream")


class RealtimeOutput(BaseAppOutput):
    text: str = Field(default="", description="The transcript so far: finished turns plus the live tail")
    turns: List[Turn] = Field(default_factory=list, description="The finished turns, in order")
    seconds: float = Field(default=0, description="Audio sent to Inworld so far, pauses included")
    partial: bool = Field(default=True, description="True while the tail may still change; False on the result")
    end_reason: str = Field(default="", description="Why the session ended, on the result")


_DONE = object()


class _Ended:
    """The session is over though the caller is still there: the transcript so far is the result."""

    def __init__(self, reason: str):
        self.reason = reason


class App(BaseApp):
    """Inworld STT app implementation."""

    async def setup(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        get_api_key()
        self._socket: Optional[Socket] = None
        self.logger.info("Inworld STT app initialized")

    async def on_cancel(self):
        # Stop reading from the caller: the uplink loop ends and the session winds down.
        socket, self._socket = self._socket, None
        if socket is not None:
            await socket.close()
        return True

    async def run(self, input_data: AppInput) -> AppOutput:
        self.logger.info(f"Transcribing audio: {input_data.audio.path}")

        result = await speech_to_text(
            audio_path=input_data.audio.path,
            model_id="inworld/inworld-stt-1",
            language=input_data.language,
            include_word_timestamps=input_data.include_word_timestamps,
            logger=self.logger,
        )

        transcription = result.get("transcription", {})
        transcript = transcription.get("transcript", "")
        words = transcription.get("wordTimestamps")

        # Get duration from usage or word timestamps, fallback to ffprobe
        usage = result.get("usage", {})
        duration_ms = usage.get("transcribedAudioMs", 0)
        duration_seconds = duration_ms / 1000.0 if duration_ms else 0.0

        if duration_seconds == 0.0 and words:
            last_word = words[-1]
            duration_seconds = last_word.get("endTimeMs", 0) / 1000.0

        if duration_seconds == 0.0:
            duration_seconds = get_audio_duration(input_data.audio.path, self.logger)

        self.logger.info(f"Transcription: {len(transcript)} chars, {duration_seconds:.2f}s audio")

        return AppOutput(
            text=transcript,
            words=words,
            output_meta=OutputMeta(
                inputs=[AudioMeta(
                    seconds=duration_seconds,
                    extra={"model": "inworld/inworld-stt-1"},
                )],
                outputs=[],
            ),
        )

    async def realtime(self, input_data: RealtimeInput, socket: Socket) -> AsyncGenerator[RealtimeOutput, None]:
        """Transcribe a microphone while it speaks."""
        import websockets

        live = Live(socket, input_data, RealtimeOutput)
        self._socket = socket
        transcript = _Transcript()
        flushed = asyncio.Event()               # Inworld has said everything it had
        snapshots: "asyncio.Queue[Any]" = asyncio.Queue()
        state: Dict[str, Any] = {
            "sent": 0,                          # bytes of audio sent to Inworld, silence included
            "heard": 0,                         # bytes of it that came from the caller
            "audio_at": time.monotonic(),       # when the caller last sent audio
            "spoke_at": time.monotonic(),       # when Inworld last heard speech
            "caller_gone": False,
            "answered": False,                  # Inworld has answered: the stream was accepted
            "reported": 0.0,                    # seconds Inworld says it transcribed
        }
        inworld_error: Dict[str, str] = {}

        inworld = await websockets.connect(
            REALTIME_URL,
            additional_headers={"Authorization": get_auth_header()},
            max_size=None,
        )
        # The config is the first frame and gets no answer of its own: a config
        # Inworld rejects closes the socket.
        await inworld.send(json.dumps({"transcribeConfig": _config(input_data)}))

        async def push() -> None:
            if not socket.closed:                 # the result is yielded after the caller has gone
                await live.send(text=transcript.text)

        async def send_audio(pcm: bytes) -> None:
            state["sent"] += len(pcm)
            await inworld.send(json.dumps({"audioChunk": {"content": base64.b64encode(pcm).decode()}}))

        async def uplink() -> None:
            pending = bytearray()
            async for update in live:
                if update.field != "audio":
                    continue                      # an ordinary field; Live already applied it
                state["heard"] += len(update.value)
                state["audio_at"] = time.monotonic()
                pending += update.value
                if len(pending) >= CHUNK_BYTES:
                    chunk, pending = bytes(pending), bytearray()
                    await send_audio(chunk)
            # The caller closed. Send what is left and close the stream:
            # Inworld answers with the last turn and its usage, then hangs up.
            state["caller_gone"] = True
            try:
                if pending:
                    await send_audio(bytes(pending))
                await inworld.send(json.dumps({"closeStream": {}}))
                await asyncio.wait_for(flushed.wait(), FLUSH_SECONDS)
            except Exception:  # noqa: BLE001 - Inworld is gone or slow; what was finished is the result
                pass
            snapshots.put_nowait(_DONE)

        async def fill_silence() -> None:
            # A microphone that gates silence sends nothing in a pause, and
            # Inworld ends a turn on the silence it hears.
            while not state["caller_gone"]:
                await asyncio.sleep(len(SILENCE) / BYTES_PER_SECOND)
                if state["caller_gone"] or time.monotonic() - state["audio_at"] < GAP_SECONDS:
                    continue
                await send_audio(SILENCE)

        async def downlink() -> None:
            try:
                async for message in inworld:
                    if isinstance(message, bytes):
                        continue
                    event = json.loads(message)
                    if "error" in event:
                        err = event.get("error") or {}
                        text = err.get("message") if isinstance(err, dict) else str(err)
                        inworld_error["message"] = f"Inworld: {text or json.dumps(event)}"
                        self.logger.warning("%s", inworld_error["message"])
                        await live.error(inworld_error["message"])
                        continue
                    state["answered"] = True
                    result = event.get("result") or {}
                    transcription = result.get("transcription")
                    usage = result.get("usage")
                    if transcription is not None:
                        if (transcription.get("transcript") or "").strip():
                            state["spoke_at"] = time.monotonic()
                        changed, finished = transcript.heard(transcription)
                        if changed:
                            await push()
                        if finished:
                            snapshots.put_nowait(None)
                    elif usage is not None:
                        ms = usage.get("transcribedAudioMs") or usage.get("transcribed_audio_ms") or 0
                        state["reported"] = float(ms) / 1000
                    elif "speechStarted" in result or "speech_started" in result:
                        state["spoke_at"] = time.monotonic()
                    elif not ("speechStopped" in result or "speech_stopped" in result):
                        self.logger.info("inworld event: %s", message[:300])
            except asyncio.CancelledError:
                raise
            except Exception as err:  # noqa: BLE001 - a closed socket ends the session, it does not fail it
                inworld_error.setdefault("message", f"Inworld closed the stream: {err}")
            flushed.set()
            if state["caller_gone"]:
                return                              # the caller closed first; the result is on its way
            # Inworld is gone. A stream it never answered was refused, which is
            # a failure; one it answered ends like any other and is billed.
            reason = inworld_error.get("message") or "Inworld closed the stream"
            snapshots.put_nowait(_Ended(reason) if state["answered"] else RuntimeError(reason))

        async def idle_watch() -> None:
            while True:
                limit = input_data.idle_minutes * 60      # read each time: it can change mid-stream
                await asyncio.sleep(min(5.0, limit / 4) if limit > 0 else 5.0)
                if limit <= 0 or time.monotonic() - state["spoke_at"] < limit:
                    continue
                minutes = f"{input_data.idle_minutes:g} minute{'' if input_data.idle_minutes == 1 else 's'}"
                reason = f"ended after {minutes} with nobody speaking"
                self.logger.info("%s", reason)
                await live.error(reason)
                snapshots.put_nowait(_Ended(reason))
                return

        def snapshot(partial: bool = True) -> RealtimeOutput:
            return RealtimeOutput(
                text=transcript.text,
                turns=list(transcript.turns),
                seconds=round(state["sent"] / BYTES_PER_SECOND, 3),
                partial=partial,
            )

        await push()                                # the first frame: the app is there
        end_reason = "the caller closed the session"
        tasks = [asyncio.create_task(job()) for job in (uplink, downlink, fill_silence, idle_watch)]
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
            await inworld.close()

        # Whatever was still interim is the last of the transcript.
        transcript.finish()
        sent = state["sent"] / BYTES_PER_SECOND
        billed = state["reported"] or sent          # Inworld's own count when the stream closed cleanly
        result = snapshot(partial=False)
        result.end_reason = end_reason
        result.output_meta = OutputMeta(
            inputs=[
                AudioMeta(
                    seconds=round(billed, 3),
                    sample_rate=SAMPLE_RATE,
                    extra={"model": MODEL, "caller_seconds": round(state["heard"] / BYTES_PER_SECOND, 3)},
                )
            ],
            outputs=[],
        )
        self.logger.info(
            "session ended (%s): %.1fs sent, %.1fs of it from the caller, Inworld counted %.1fs, %d turns",
            end_reason, sent, state["heard"] / BYTES_PER_SECOND, state["reported"], len(transcript.turns),
        )
        yield result


class _Transcript:
    """The transcript as Inworld reports it: an interim transcript is the turn
    in progress as heard so far and replaces the one before; a final one
    finishes the turn."""

    def __init__(self) -> None:
        self.turns: List[Turn] = []
        self._interim = ""
        self._sent = ""

    @property
    def text(self) -> str:
        return " ".join(p for p in [t.text for t in self.turns] + [self._interim] if p).strip()

    def heard(self, transcription: Dict[str, Any]):
        """Applies a transcript; (the text changed, a turn finished)."""
        text = (transcription.get("transcript") or "").strip()
        final = transcription.get("isFinal", transcription.get("is_final", False))
        finished = False
        if not final:
            self._interim = text
        else:
            self._interim = ""
            if text:
                words = transcription.get("wordTimestamps") or transcription.get("word_timestamps") or []
                at = self.turns[-1].end if self.turns else 0.0
                start, end = at, at
                if words:
                    start = float(words[0].get("startTimeMs") or words[0].get("start_time_ms") or 0) / 1000
                    end = float(words[-1].get("endTimeMs") or words[-1].get("end_time_ms") or 0) / 1000
                self.turns.append(Turn(text=text, start=round(start, 3), end=round(end, 3)))
                finished = True
        changed = self.text != self._sent
        self._sent = self.text
        return changed, finished

    def finish(self) -> None:
        """Closes the turn in progress, if there is one."""
        if self._interim:
            at = self.turns[-1].end if self.turns else 0.0
            self.turns.append(Turn(text=self._interim, start=at, end=at))
            self._interim = ""


def _config(input_data: RealtimeInput) -> Dict[str, Any]:
    config: Dict[str, Any] = {
        "modelId": MODEL,
        "audioEncoding": "LINEAR16",
        "sampleRateHertz": SAMPLE_RATE,
        "numberOfChannels": 1,
        "includeWordTimestamps": True,
    }
    if input_data.language:
        config["language"] = input_data.language
    if input_data.keyterms:
        config["prompts"] = list(input_data.keyterms)
    if input_data.silence_ms is not None:
        config["inworldSttV1Config"] = {"maxTurnSilence": input_data.silence_ms}
    return config

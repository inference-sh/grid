"""
ElevenLabs Speech to Text (Scribe)

High-accuracy speech transcription with speaker diarization
and audio event detection. Supports 90+ languages.

`run` takes an audio file. `realtime` is a live function: the caller streams
microphone audio and reads the transcript as it forms. The app relays Scribe
v2 Realtime (``wss://api.elevenlabs.io/v1/speech-to-text/realtime``), which
answers with partial text and commits a segment when the speaker pauses.

    caller -> app   <binary PCM s16le mono 16 kHz>   an item of RealtimeInput.audio (a frame every ~20 ms)
    app -> caller   {"text": ""}                     the first frame: ElevenLabs has started the session
    app -> caller   {"text": "what's the wea"}       the transcript so far, its tail still changing
    caller closes  ->  the tail is committed and the function yields its result

The socket is the low-latency channel; the yields are the task's output. Each
committed segment yields a cumulative snapshot, and the yield after the socket
closes is the result and carries ``output_meta``.

ElevenLabs reads its options from the connection URL, so the ordinary inputs
of `realtime` are fixed once the session is open; only ``idle_minutes`` can
change mid-stream. ElevenLabs closes a session that gets nothing for 15
seconds, so while the caller sends nothing the app sends silence.
"""

import asyncio
import base64
import json
import logging
import time
from typing import Annotated, Any, AsyncGenerator, Dict, List, Literal, Optional, Tuple
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

from .elevenlabs_helper import speech_to_text, get_api_key, get_audio_duration

REALTIME_MODEL = "scribe_v2_realtime"
REALTIME_URL = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"
SAMPLE_RATE = 16000
BYTES_PER_SECOND = SAMPLE_RATE * 2
CHUNK_BYTES = BYTES_PER_SECOND // 10    # ElevenLabs asks for chunks of 0.1 to 1 second
GAP_SECONDS = 0.5                       # the caller has gone quiet: fill with silence from here on
FLUSH_SECONDS = 3.0                     # how long ElevenLabs gets to commit the tail after the caller closes
SILENCE = bytes(CHUNK_BYTES)
Keyterm = Annotated[str, Field(max_length=20)]


class AppInput(BaseAppInput):
    """Input schema for ElevenLabs STT."""

    audio: File = Field(
        description="Audio file to transcribe (MP3, WAV, FLAC, OGG, AAC, M4A).",
    )
    model: Literal["scribe_v1", "scribe_v2"] = Field(
        default="scribe_v2",
        description="Model version. scribe_v2 is latest with improved accuracy.",
    )
    language_code: Optional[str] = Field(
        default=None,
        description="Language code (e.g., 'eng', 'spa', 'fra'). Leave empty for auto-detection.",
    )
    diarize: bool = Field(
        default=False,
        description="Enable speaker diarization to identify who is speaking.",
    )
    tag_audio_events: bool = Field(
        default=False,
        description="Tag audio events like laughter, applause, music, etc.",
    )


class AppOutput(BaseAppOutput):
    """Output schema for ElevenLabs STT."""
    text: str = Field(description="Full transcription text")
    language_code: Optional[str] = Field(default=None, description="Detected language code")
    language_probability: Optional[float] = Field(default=None, description="Language detection confidence")
    words: Optional[List[dict]] = Field(default=None, description="Word-level timestamps and speaker info")


class RealtimeInput(BaseAppInput):
    audio: Stream[PCM16(SAMPLE_RATE)] = Field(description="Microphone audio, a frame every 20 ms or so")
    language_code: Optional[str] = Field(
        default=None,
        description="Language code (e.g., 'en', 'es', 'fra'). Leave empty for auto-detection.",
    )
    keyterms: List[Keyterm] = Field(
        default_factory=list,
        max_length=50,
        description="Names and terms the transcript should prefer, up to 50 of at most 20 characters each. ElevenLabs charges 20% more for a session that uses them.",
    )
    silence_ms: int = Field(default=1500, ge=300, le=3000, description="Silence that commits a segment, in ms")
    idle_minutes: float = Field(
        default=2.0,
        ge=0,
        le=60,
        description="End the session after this long with nobody speaking (0: never). ElevenLabs bills every minute of audio it is sent, silent or not.",
    )


class Segment(BaseModel):
    """A committed stretch of speech. Its times are seconds from the start of the stream."""

    text: str = Field(description="What was said")
    start: float = Field(default=0, description="Seconds from the start of the stream")
    end: float = Field(default=0, description="Seconds from the start of the stream")


class RealtimeOutput(BaseAppOutput):
    text: str = Field(default="", description="The transcript so far: committed segments plus the live tail")
    segments: List[Segment] = Field(default_factory=list, description="The committed segments, in order")
    language_code: Optional[str] = Field(default=None, description="The language ElevenLabs detected")
    seconds: float = Field(default=0, description="Audio sent to ElevenLabs so far, pauses included")
    partial: bool = Field(default=True, description="True while the tail may still change; False on the result")
    end_reason: str = Field(default="", description="Why the session ended, on the result")


_DONE = object()


class _Ended:
    """The session is over though the caller is still there: the transcript so far is the result."""

    def __init__(self, reason: str):
        self.reason = reason


class App(BaseApp):
    """ElevenLabs STT app implementation."""

    async def setup(self):
        """Initialize the application."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.api_key = get_api_key()
        self._socket: Optional[Socket] = None
        self.logger.info("ElevenLabs STT app initialized")

    async def on_cancel(self):
        # Stop reading from the caller: the uplink loop ends and the session winds down.
        socket, self._socket = self._socket, None
        if socket is not None:
            await socket.close()
        return True

    async def run(self, input_data: AppInput) -> AppOutput:
        """Transcribe audio to text."""
        self.logger.info(f"Transcribing audio: {input_data.audio.path}")
        self.logger.info(f"Model: {input_data.model}, Diarize: {input_data.diarize}")

        result = speech_to_text(
            audio=input_data.audio.path,
            model_id=input_data.model,
            language_code=input_data.language_code,
            diarize=input_data.diarize,
            tag_audio_events=input_data.tag_audio_events,
            logger=self.logger,
        )

        # Get duration from API response word timestamps, fallback to ffprobe
        duration_seconds = 0.0
        words = result.get("words", [])
        if words and len(words) > 0:
            # Use end time of last word as duration
            duration_seconds = float(words[-1].get("end", 0))
            self.logger.info(f"Duration from API: {duration_seconds:.2f}s")
        if duration_seconds == 0.0:
            duration_seconds = get_audio_duration(input_data.audio.path, self.logger)

        return AppOutput(
            text=result.get("text", ""),
            language_code=result.get("language_code"),
            language_probability=result.get("language_probability"),
            words=words,
            output_meta=OutputMeta(
                inputs=[AudioMeta(
                    seconds=duration_seconds,
                    extra={"model": input_data.model}
                )],
                outputs=[]
            )
        )

    async def realtime(self, input_data: RealtimeInput, socket: Socket) -> AsyncGenerator[RealtimeOutput, None]:
        """Transcribe a microphone while it speaks."""
        import websockets

        live = Live(socket, input_data, RealtimeOutput)
        self._socket = socket
        transcript = _Transcript()
        ready = asyncio.Event()                 # ElevenLabs started the session
        flushed = asyncio.Event()               # ElevenLabs committed the tail
        snapshots: "asyncio.Queue[Any]" = asyncio.Queue()
        state: Dict[str, Any] = {
            "sent": 0,                          # bytes of audio sent to ElevenLabs, silence included
            "heard": 0,                         # bytes of it that came from the caller
            "audio_at": time.monotonic(),       # when the caller last sent audio
            "spoke_at": time.monotonic(),       # when ElevenLabs last heard words
            "caller_gone": False,
        }
        eleven_error: Dict[str, str] = {}

        eleven = await websockets.connect(
            _realtime_url(input_data),
            additional_headers={"xi-api-key": self.api_key},
            max_size=None,
        )

        async def push() -> None:
            if not socket.closed:                 # the result is yielded after the caller has gone
                await live.send(text=transcript.text)

        async def send_audio(pcm: bytes, commit: bool = False) -> None:
            state["sent"] += len(pcm)
            await eleven.send(json.dumps({
                "message_type": "input_audio_chunk",
                "audio_base_64": base64.b64encode(pcm).decode(),
                "commit": commit,
                "sample_rate": SAMPLE_RATE,
            }))

        async def uplink() -> None:
            await ready.wait()
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
            # The caller closed. Send what is left and commit it: ElevenLabs
            # answers a commit with the final text of the segment.
            state["caller_gone"] = True
            try:
                if pending:
                    await send_audio(bytes(pending))
                await send_audio(b"", commit=True)
                await asyncio.wait_for(flushed.wait(), FLUSH_SECONDS)
            except Exception:  # noqa: BLE001 - ElevenLabs is gone or slow; what was committed is the result
                pass
            snapshots.put_nowait(_DONE)

        async def fill_silence() -> None:
            # A microphone that gates silence sends nothing in a pause.
            # ElevenLabs needs to hear the pause to commit the segment, and
            # closes a session it hears nothing from for 15 seconds.
            await ready.wait()
            while not state["caller_gone"]:
                await asyncio.sleep(len(SILENCE) / BYTES_PER_SECOND)
                if state["caller_gone"] or time.monotonic() - state["audio_at"] < GAP_SECONDS:
                    continue
                await send_audio(SILENCE)

        async def downlink() -> None:
            try:
                async for message in eleven:
                    if isinstance(message, bytes):
                        continue
                    event = json.loads(message)
                    kind = event.get("message_type", "")
                    if kind == "session_started":
                        if not ready.is_set():
                            ready.set()
                            await push()              # the first frame: the app is there
                    elif kind == "partial_transcript":
                        if (event.get("text") or "").strip():
                            state["spoke_at"] = time.monotonic()
                        if transcript.partial(event.get("text") or ""):
                            await push()
                    elif kind in ("committed_transcript", "final_transcript"):
                        if transcript.committed(event.get("text") or ""):
                            await push()
                            snapshots.put_nowait(None)
                    elif kind in ("committed_transcript_with_timestamps", "final_transcript_with_timestamps"):
                        # The second event of a commit; after the caller closed it is the last one.
                        transcript.timed(event)
                        if state["caller_gone"]:
                            flushed.set()
                    elif kind == "warning":
                        self.logger.warning("ElevenLabs: %s", event.get("warning"))
                    elif "error" in event:
                        # commit_throttled and the like leave the session up;
                        # anything fatal and ElevenLabs closes it. Tell the caller.
                        eleven_error["message"] = f"ElevenLabs: {event.get('error') or kind}"
                        self.logger.warning("%s (%s)", eleven_error["message"], kind)
                        await live.error(eleven_error["message"])
                        if state["caller_gone"]:
                            flushed.set()
                    else:
                        self.logger.info("elevenlabs event %s: %s", kind, message[:300])
            except asyncio.CancelledError:
                raise
            except Exception as err:  # noqa: BLE001 - a closed socket ends the session, it does not fail it
                eleven_error.setdefault("message", f"ElevenLabs closed the session: {err}")
            flushed.set()
            if state["caller_gone"]:
                return                              # the caller closed first; the result is on its way
            # ElevenLabs is gone. A session it never started is a failure; one
            # it started ends like any other and is billed.
            reason = eleven_error.get("message") or "ElevenLabs closed the session"
            snapshots.put_nowait(_Ended(reason) if ready.is_set() else RuntimeError(reason))

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
                segments=list(transcript.segments),
                language_code=transcript.language_code,
                seconds=round(state["sent"] / BYTES_PER_SECOND, 3),
                partial=partial,
            )

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
            await eleven.close()

        # Whatever was still uncommitted is the last of the transcript.
        transcript.committed(transcript.tail)
        seconds = state["sent"] / BYTES_PER_SECOND
        result = snapshot(partial=False)
        result.end_reason = end_reason
        # ElevenLabs reports no usage: it bills the audio it was sent, and
        # 20% more when the session used keyterms.
        result.output_meta = OutputMeta(
            inputs=[
                AudioMeta(
                    seconds=round(seconds, 3),
                    sample_rate=SAMPLE_RATE,
                    extra={
                        "model": REALTIME_MODEL,
                        "keyterms": bool(input_data.keyterms),
                        "caller_seconds": round(state["heard"] / BYTES_PER_SECOND, 3),
                    },
                )
            ],
            outputs=[],
        )
        self.logger.info(
            "session ended (%s): %.1fs sent, %.1fs of it from the caller, %d segments",
            end_reason, seconds, state["heard"] / BYTES_PER_SECOND, len(transcript.segments),
        )
        yield result


class _Transcript:
    """The transcript as ElevenLabs reports it: a partial is the uncommitted
    tail as heard so far and replaces the one before; a commit settles it. The
    timings of a commit arrive in a second event that repeats its text."""

    def __init__(self) -> None:
        self.segments: List[Segment] = []
        self.language_code: Optional[str] = None
        self.tail = ""

    @property
    def text(self) -> str:
        return " ".join(p for p in [s.text for s in self.segments] + [self.tail] if p).strip()

    def partial(self, text: str) -> bool:
        """Applies a partial; True when the tail changed."""
        text = text.strip()
        changed, self.tail = text != self.tail, text
        return changed

    def committed(self, text: str) -> bool:
        """Settles a segment; True when it had words."""
        text = text.strip()
        if not text:
            return False                    # a pause committed with nothing said in it
        self.tail = ""
        at = self.segments[-1].end if self.segments else 0.0
        self.segments.append(Segment(text=text, start=at, end=at))
        return True

    def timed(self, event: Dict[str, Any]) -> None:
        """Puts the timings of a commit on its segment."""
        self.language_code = event.get("language_code") or self.language_code
        times = [
            (float(w.get("start") or 0), float(w.get("end") or 0))
            for w in event.get("words") or []
            if isinstance(w, dict) and w.get("type", "word") == "word"
        ]
        text = (event.get("text") or "").strip()
        if not times or not text:
            return
        for segment in reversed(self.segments):
            if segment.text == text:
                segment.start, segment.end = round(times[0][0], 3), round(times[-1][1], 3)
                return


def _realtime_url(input_data: RealtimeInput) -> str:
    query: List[Tuple[str, Any]] = [
        ("model_id", REALTIME_MODEL),
        ("audio_format", f"pcm_{SAMPLE_RATE}"),
        ("commit_strategy", "vad"),
        ("vad_silence_threshold_secs", input_data.silence_ms / 1000),
        ("include_timestamps", "true"),
    ]
    if input_data.language_code:
        query.append(("language_code", input_data.language_code))
    else:
        query.append(("include_language_detection", "true"))
    query.extend(("keyterms", term) for term in input_data.keyterms)
    return f"{REALTIME_URL}?{urlencode(query)}"

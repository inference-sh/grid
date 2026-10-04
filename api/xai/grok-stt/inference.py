"""Grok Speech to Text: transcribe a recording, or a microphone while it speaks.

`run` takes an audio file and returns its transcript with word timings.

`realtime` is a live function: the caller streams microphone audio and reads
the transcript as it forms. The app relays xAI's streaming endpoint
(``wss://api.x.ai/v1/stt``), which takes raw PCM frames and answers with
interim text that it later locks, a few seconds at a time, and settles into an
utterance when the speaker pauses.

    caller -> app   <binary PCM s16le mono 16 kHz>   an item of RealtimeInput.audio (a frame every ~20 ms)
    app -> caller   {"text": ""}                     the first frame: xAI has accepted the stream
    app -> caller   {"text": "what's the wea"}       the transcript so far, its tail still changing
    caller closes  ->  xAI flushes what it holds and the function yields its result

The socket is the low-latency channel; the yields are the task's output. Each
settled utterance yields a cumulative snapshot, and the yield after the socket
closes is the result and carries ``output_meta``.

xAI reads its options from the connection URL, so the ordinary inputs of
`realtime` are fixed once the stream is open; only ``idle_minutes`` can change
mid-stream. While the caller sends nothing the app sends silence, so a pause
neither stalls xAI's end-of-utterance detection nor times the stream out.
"""

import asyncio
import json
import logging
import mimetypes
import os
import time
from collections import Counter
from typing import Annotated, Any, AsyncGenerator, Dict, List, Literal, Optional, Tuple
from urllib.parse import urlencode

import httpx
from pydantic import BaseModel, Field

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

MODEL = "grok-voice-transcribe-2.0"
REST_URL = "https://api.x.ai/v1/stt"
STREAM_URL = "wss://api.x.ai/v1/stt"
SAMPLE_RATE = 16000                 # the model's native rate: nothing is resampled on xAI's side
BYTES_PER_SECOND = SAMPLE_RATE * 2
GAP_SECONDS = 0.5                   # the caller has gone quiet: fill with silence from here on
FLUSH_SECONDS = 3.0                 # how long xAI gets to hand over the tail after the caller closes
SILENCE = bytes(BYTES_PER_SECOND // 10)

# The languages xAI formats numbers, currencies and units for (speech in any
# of them is transcribed whether or not one is named).
Language = Literal[
    "ar", "cs", "da", "nl", "en", "fil", "fr", "de", "hi", "id", "it", "ja", "ko",
    "mk", "ms", "fa", "pl", "pt", "ro", "ru", "es", "sv", "th", "tr", "vi",
]
Keyterm = Annotated[str, Field(max_length=50)]


class Word(BaseModel):
    text: str
    start: float = Field(description="Seconds from the start of the audio")
    end: float = Field(description="Seconds from the start of the audio")
    confidence: Optional[float] = Field(default=None, description="0 to 1, when xAI reports it")
    speaker: Optional[int] = Field(default=None, description="Speaker number from 0, with diarize")


class AppInput(BaseAppInput):
    audio: File = Field(description="The recording: wav, mp3, ogg, opus, flac, aac, m4a, mp4 or mkv, up to 500 MB")
    language: Optional[Language] = Field(
        default=None,
        description="What is spoken. Speech is transcribed in any supported language without it; naming it writes numbers, currencies and units the way they are written (\"$167,983.15\").",
    )
    diarize: bool = Field(default=False, description="Tell speakers apart: each word carries a speaker number")
    keyterms: List[Keyterm] = Field(
        default_factory=list,
        max_length=100,
        description="Names and terms the transcript should prefer, up to 100 of at most 50 characters each",
    )
    filler_words: bool = Field(default=False, description="Keep fillers such as \"um\" and \"uh\"")


class AppOutput(BaseAppOutput):
    text: str = Field(description="The transcript")
    language: str = Field(default="", description="The language xAI detected, such as en or es-mx")
    duration: float = Field(default=0, description="Length of the audio, in seconds")
    words: List[Word] = Field(default_factory=list, description="Every word with its timing")


class RealtimeInput(BaseAppInput):
    audio: Stream[PCM16(SAMPLE_RATE)] = Field(description="Microphone audio, a frame every 20 ms or so")
    language: Optional[Language] = Field(
        default=None,
        description="What is spoken. Speech is transcribed in any supported language without it; naming it writes numbers, currencies and units the way they are written.",
    )
    diarize: bool = Field(default=False, description="Tell speakers apart: each utterance carries a speaker number")
    keyterms: List[Keyterm] = Field(
        default_factory=list,
        max_length=100,
        description="Names and terms the transcript should prefer, up to 100 of at most 50 characters each",
    )
    filler_words: bool = Field(default=False, description="Keep fillers such as \"um\" and \"uh\"")
    silence_ms: int = Field(default=400, ge=0, le=5000, description="Silence that settles an utterance, in ms")
    idle_minutes: float = Field(
        default=2.0,
        ge=0,
        le=60,
        description="End the session after this long with nobody speaking (0: never). xAI bills every minute the stream is open, silent or not.",
    )


class Utterance(BaseModel):
    """A settled stretch of speech. Its times are seconds from the start of the stream."""

    text: str = Field(description="What was said")
    start: float = Field(description="Seconds from the start of the stream")
    end: float = Field(description="Seconds from the start of the stream")
    speaker: Optional[int] = Field(default=None, description="Who spoke most of it, with diarize")


class RealtimeOutput(BaseAppOutput):
    text: str = Field(default="", description="The transcript so far: settled utterances plus the live tail")
    utterances: List[Utterance] = Field(default_factory=list, description="The settled utterances, in order")
    seconds: float = Field(default=0, description="Audio sent to xAI so far, pauses included")
    partial: bool = Field(default=True, description="True while the tail may still change; False on the result")
    end_reason: str = Field(default="", description="Why the session ended, on the result")


_DONE = object()


class _Ended:
    """The session is over though the caller is still there: the transcript so far is the result."""

    def __init__(self, reason: str):
        self.reason = reason


class App(BaseApp):
    async def setup(self, metadata):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        self.api_key = os.environ.get("XAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("XAI_API_KEY environment variable is required")
        self._socket: Optional[Socket] = None

    async def on_cancel(self):
        # Stop reading from the caller: the uplink loop ends and the session winds down.
        socket, self._socket = self._socket, None
        if socket is not None:
            await socket.close()
        return True

    async def run(self, input_data: AppInput) -> AppOutput:
        """Transcribe a recording."""
        path = input_data.audio.path
        # xAI reads the options before the file, so `file` goes last: httpx
        # writes `data` first, then `files`.
        fields: Dict[str, Any] = {
            "model": MODEL,
            "diarize": _flag(input_data.diarize),
            "filler_words": _flag(input_data.filler_words),
        }
        if input_data.language:
            fields["language"] = input_data.language
            fields["format"] = "true"
        if input_data.keyterms:
            fields["keyterm"] = list(input_data.keyterms)

        self.logger.info("transcribing %s (%.1f MB)", os.path.basename(path), os.path.getsize(path) / 1e6)
        content_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
        async with httpx.AsyncClient(timeout=httpx.Timeout(900, connect=30)) as client:
            with open(path, "rb") as audio:
                response = await client.post(
                    REST_URL,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    data=fields,
                    files={"file": (os.path.basename(path), audio, content_type)},
                )
        if response.status_code != 200:
            raise RuntimeError(f"xAI answered {response.status_code}: {response.text[:500]}")
        data = response.json()

        words = [_word(item) for item in data.get("words") or [] if isinstance(item, dict)]
        duration = float(data.get("duration") or (words[-1].end if words else 0))
        self.logger.info("transcribed %.1fs of audio: %d words", duration, len(words))
        return AppOutput(
            text=data.get("text") or "",
            language=data.get("language") or "",
            duration=duration,
            words=words,
            output_meta=OutputMeta(inputs=[AudioMeta(seconds=duration, extra={"model": MODEL})], outputs=[]),
        )

    async def realtime(self, input_data: RealtimeInput, socket: Socket) -> AsyncGenerator[RealtimeOutput, None]:
        """Transcribe a microphone while it speaks."""
        import websockets

        live = Live(socket, input_data, RealtimeOutput)
        self._socket = socket
        transcript = _Transcript()
        ready = asyncio.Event()                 # xAI accepted the stream
        flushed = asyncio.Event()               # xAI has said everything it had
        snapshots: "asyncio.Queue[Any]" = asyncio.Queue()
        state: Dict[str, Any] = {
            "sent": 0,                          # bytes of audio sent to xAI, silence included
            "heard": 0,                         # bytes of it that came from the caller
            "audio_at": time.monotonic(),       # when the caller last sent audio
            "spoke_at": time.monotonic(),       # when xAI last heard words
            "caller_gone": False,
            "reported": 0.0,                    # seconds xAI says it processed
        }
        xai_error: Dict[str, str] = {}

        xai = await websockets.connect(
            _stream_url(input_data),
            additional_headers={"Authorization": f"Bearer {self.api_key}"},
            max_size=None,
        )

        async def push() -> None:
            if not socket.closed:                 # the result is yielded after the caller has gone
                await live.send(text=transcript.text)

        async def uplink() -> None:
            await ready.wait()
            async for update in live:
                if update.field != "audio":
                    continue                      # an ordinary field; Live already applied it
                state["heard"] += len(update.value)
                state["sent"] += len(update.value)
                state["audio_at"] = time.monotonic()
                await xai.send(update.value)
            # The caller closed. xAI still holds the tail: ask for it and give it a moment.
            state["caller_gone"] = True
            try:
                await xai.send(json.dumps({"type": "audio.done"}))
                await asyncio.wait_for(flushed.wait(), FLUSH_SECONDS)
            except Exception:  # noqa: BLE001 - xAI is gone or slow; what was settled is the result
                pass
            snapshots.put_nowait(_DONE)

        async def fill_silence() -> None:
            # A microphone that gates silence sends nothing in a pause. xAI
            # needs to hear the pause to settle the utterance, and an idle
            # stream may be closed, so the pause is sent as silence.
            await ready.wait()
            while not state["caller_gone"]:
                await asyncio.sleep(len(SILENCE) / BYTES_PER_SECOND)
                if state["caller_gone"] or time.monotonic() - state["audio_at"] < GAP_SECONDS:
                    continue
                state["sent"] += len(SILENCE)
                await xai.send(SILENCE)

        async def downlink() -> None:
            try:
                async for message in xai:
                    if isinstance(message, bytes):
                        continue
                    event = json.loads(message)
                    kind = event.get("type", "")
                    if kind == "transcript.created":
                        if not ready.is_set():
                            ready.set()
                            await push()              # the first frame: the app is there
                    elif kind == "transcript.partial":
                        if (event.get("text") or "").strip():
                            state["spoke_at"] = time.monotonic()
                        changed, settled = transcript.heard(event)
                        if changed:
                            await push()
                        if settled:
                            snapshots.put_nowait(None)
                    elif kind == "transcript.done":
                        state["reported"] = float(event.get("duration") or 0)
                        break
                    elif kind == "error":
                        # A message xAI could not parse leaves the stream up;
                        # anything else and xAI closes it. Tell the caller.
                        xai_error["message"] = f"xAI: {event.get('message') or json.dumps(event)}"
                        self.logger.warning("%s", xai_error["message"])
                        await live.error(xai_error["message"])
                    else:
                        self.logger.info("xai event %s: %s", kind, message[:300])
            except asyncio.CancelledError:
                raise
            except Exception as err:  # noqa: BLE001 - a closed socket ends the session, it does not fail it
                xai_error.setdefault("message", f"xAI closed the stream: {err}")
            flushed.set()
            if state["caller_gone"]:
                return                              # the caller closed first; the result is on its way
            # xAI is gone. A stream it never accepted is a failure; one it
            # accepted ends like any other and is billed.
            reason = xai_error.get("message") or "xAI closed the stream"
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
                utterances=list(transcript.utterances),
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
            await xai.close()

        # Whatever was still unsettled is the last of the transcript.
        transcript.settle()
        sent = state["sent"] / BYTES_PER_SECOND
        billed = state["reported"] or sent          # xAI's own count when the stream closed cleanly
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
            "session ended (%s): %.1fs sent, %.1fs of it from the caller, xAI counted %.1fs, %d utterances",
            end_reason, sent, state["heard"] / BYTES_PER_SECOND, state["reported"], len(transcript.utterances),
        )
        yield result


class _Transcript:
    """The transcript as xAI reports it.

    An interim result is the current stretch as heard so far and is replaced by
    the next one. A final result locks a stretch (about three seconds of
    speech). The final that ends an utterance carries the whole utterance,
    stitched, when it starts where the utterance started; otherwise it is the
    utterance's last stretch.
    """

    def __init__(self) -> None:
        self.utterances: List[Utterance] = []
        self._locked: List[Tuple[float, float, str]] = []   # (start, end, text) of the utterance in progress
        self._speakers: Counter = Counter()
        self._interim = ""
        self._sent = ""

    @property
    def text(self) -> str:
        parts = [u.text for u in self.utterances] + [text for _, _, text in self._locked] + [self._interim]
        return " ".join(p for p in parts if p).strip()

    def heard(self, event: Dict[str, Any]) -> Tuple[bool, bool]:
        """Applies a result; (the text changed, an utterance settled)."""
        text = (event.get("text") or "").strip()
        start = float(event.get("start") or 0)
        end = start + float(event.get("duration") or 0)
        settled = False
        if not event.get("is_final"):
            self._interim = text
        else:
            self._interim = ""
            for word in event.get("words") or []:
                if isinstance(word, dict) and word.get("speaker") is not None:
                    self._speakers[word["speaker"]] += 1
            if event.get("speech_final"):
                if self._locked and text and start <= self._locked[0][0] + 0.05:
                    self._locked = [(start, end, text)]         # the stitched utterance replaces its stretches
                elif text:
                    self._locked.append((start, end, text))
                settled = self.settle()
            elif text:
                self._locked.append((start, end, text))
        changed = self.text != self._sent
        self._sent = self.text
        return changed, settled

    def settle(self) -> bool:
        """Closes the utterance in progress; True when there was one."""
        if self._interim:
            last = self._locked[-1][1] if self._locked else (self.utterances[-1].end if self.utterances else 0.0)
            self._locked.append((last, last, self._interim))
            self._interim = ""
        if not self._locked:
            return False
        speaker = self._speakers.most_common(1)[0][0] if self._speakers else None
        self.utterances.append(
            Utterance(
                text=" ".join(text for _, _, text in self._locked),
                start=round(self._locked[0][0], 3),
                end=round(self._locked[-1][1], 3),
                speaker=speaker,
            )
        )
        self._locked, self._speakers = [], Counter()
        return True


def _flag(value: bool) -> str:
    return "true" if value else "false"


def _word(item: Dict[str, Any]) -> Word:
    return Word(
        text=str(item.get("text") or item.get("word") or ""),
        start=float(item.get("start") or 0),
        end=float(item.get("end") or 0),
        confidence=item.get("confidence"),
        speaker=item.get("speaker"),
    )


def _stream_url(input_data: RealtimeInput) -> str:
    query: List[Tuple[str, Any]] = [
        ("model", MODEL),
        ("sample_rate", SAMPLE_RATE),
        ("encoding", "pcm"),
        ("interim_results", "true"),
        ("endpointing", input_data.silence_ms),
        ("diarize", _flag(input_data.diarize)),
        ("filler_words", _flag(input_data.filler_words)),
    ]
    if input_data.language:
        query.append(("language", input_data.language))
    query.extend(("keyterm", term) for term in input_data.keyterms)
    return f"{STREAM_URL}?{urlencode(query)}"

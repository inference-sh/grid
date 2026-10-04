"""GPT Transcribe: transcribe a recording, or a microphone while it speaks.

`run` takes an audio file and returns its transcript (``gpt-transcribe``).

`realtime` is a live function: the caller streams microphone audio and reads
the transcript as it forms. The app relays a transcription session of OpenAI's
Realtime API (``wss://api.openai.com/v1/realtime?intent=transcription``) on
``gpt-live-transcribe``, which sends the words as it hears them and the final
text of a turn once the turn is committed. That model detects no turns of its
own, so the app does: it listens for a pause in the caller's audio and commits
there.

    caller -> app   <binary PCM s16le mono 24 kHz>   an item of RealtimeInput.audio (a frame every ~20 ms)
    caller -> app   {"keyterms": ["AC-42"]}          change an ordinary input mid-stream (a session.update)
    app -> caller   {"text": ""}                     the first frame: OpenAI has accepted the session
    app -> caller   {"text": "what's the wea"}       the transcript so far, its tail still growing
    caller closes  ->  the last turn is committed and the function yields its result

The socket is the low-latency channel; the yields are the task's output. Each
finished turn yields a cumulative snapshot, and the yield after the socket
closes is the result and carries ``output_meta``.
"""

import array
import asyncio
import base64
import json
import logging
import math
import mimetypes
import os
import sys
import time
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional, Tuple

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

FILE_MODEL = "gpt-transcribe"
LIVE_MODEL = "gpt-live-transcribe"
REST_URL = "https://api.openai.com/v1/audio/transcriptions"
REALTIME_URL = "wss://api.openai.com/v1/realtime?intent=transcription"
SAMPLE_RATE = 24000                 # the only PCM rate a Realtime session takes
BYTES_PER_SECOND = SAMPLE_RATE * 2
CHUNK_BYTES = BYTES_PER_SECOND // 10
SPEECH_RMS = 400                    # of 32768: louder than this is somebody speaking
MIN_TURN_BYTES = BYTES_PER_SECOND // 5      # a commit needs audio in the buffer
MAX_TURN_SECONDS = 30.0             # commit a turn that never pauses, so its final text arrives
FLUSH_SECONDS = 3.0                 # how long OpenAI gets to finish the last turn after the caller closes

SESSION_UPDATED = {"session.updated", "transcription_session.updated"}
DELTA = "conversation.item.input_audio_transcription.delta"
COMPLETED = "conversation.item.input_audio_transcription.completed"
FAILED = "conversation.item.input_audio_transcription.failed"
# Lifecycle chatter the app has no use for.
QUIET_EVENTS = {
    "session.created",
    "transcription_session.created",
    "conversation.item.added",
    "conversation.item.created",
    "conversation.item.done",
    "input_audio_buffer.speech_started",
    "input_audio_buffer.speech_stopped",
    "input_audio_buffer.cleared",
}


class AppInput(BaseAppInput):
    audio: File = Field(description="The recording: mp3, mp4, mpeg, mpga, m4a, wav, webm, flac or ogg, up to 25 MB")
    prompt: Optional[str] = Field(
        default=None, description="What the recording is, or where it was made: context that helps with what is said in it"
    )
    languages: List[str] = Field(
        default_factory=list,
        description="The languages expected, as ISO 639-1 codes such as en or fr. Left empty the language is detected.",
    )
    keyterms: List[str] = Field(
        default_factory=list,
        description="Product names, acronyms and other literal terms that may be said. Each on one line, without < or >.",
    )


class AppOutput(BaseAppOutput):
    text: str = Field(description="The transcript")
    languages: List[str] = Field(default_factory=list, description="The languages OpenAI detected; empty when it could not tell")
    duration: float = Field(default=0, description="Length of the audio, in seconds")


class RealtimeInput(BaseAppInput):
    audio: Stream[PCM16(SAMPLE_RATE)] = Field(description="Microphone audio, a frame every 20 ms or so")
    prompt: Optional[str] = Field(
        default=None, description="What is being recorded, or where: context that helps with what is said"
    )
    languages: List[str] = Field(
        default_factory=list,
        description="The languages expected, as ISO 639-1 codes such as en or fr. Left empty the language is detected.",
    )
    keyterms: List[str] = Field(
        default_factory=list,
        description="Product names, acronyms and other literal terms that may be said. Each on one line, without < or >.",
    )
    delay: Optional[Literal["minimal", "low", "medium", "high", "xhigh"]] = Field(
        default=None,
        description="How long the model listens before it writes: lower shows words sooner, higher gets more of them right. Left empty OpenAI decides.",
    )
    silence_ms: int = Field(default=700, ge=200, le=5000, description="Silence that ends a turn, in ms")
    idle_minutes: float = Field(
        default=2.0,
        ge=0,
        le=60,
        description="End the session after this long with nobody speaking (0: never). OpenAI bills every minute of audio it is sent, silent or not.",
    )


class Turn(BaseModel):
    """A finished turn. Its times are where its audio began and ended in the stream, in seconds."""

    text: str = Field(description="What was said")
    start: float = Field(default=0, description="Seconds of audio before the turn")
    end: float = Field(default=0, description="Seconds of audio up to the end of the turn")


class RealtimeOutput(BaseAppOutput):
    text: str = Field(default="", description="The transcript so far: finished turns plus the live tail")
    turns: List[Turn] = Field(default_factory=list, description="The finished turns, in order")
    seconds: float = Field(default=0, description="Audio sent to OpenAI so far")
    partial: bool = Field(default=True, description="True while the tail may still change; False on the result")
    end_reason: str = Field(default="", description="Why the session ended, on the result")


# Ordinary input fields that change the OpenAI session when the caller changes them mid-stream.
SESSION_FIELDS = {"prompt", "languages", "keyterms", "delay"}

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
        self.api_key = os.environ.get("OPENAI_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_KEY environment variable is required")
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
        fields: Dict[str, Any] = {"model": FILE_MODEL}
        if input_data.prompt:
            fields["prompt"] = input_data.prompt
        if input_data.languages:
            fields["languages[]"] = list(input_data.languages)
        if input_data.keyterms:
            fields["keywords[]"] = list(input_data.keyterms)

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
            raise RuntimeError(f"OpenAI answered {response.status_code}: {response.text[:500]}")
        data = response.json()

        # OpenAI bills this model by the minute. It reports the length when
        # its usage is counted in seconds; otherwise the file is measured here.
        usage = data.get("usage") or {}
        duration = float(usage.get("seconds") or data.get("duration") or 0) or _audio_seconds(path)
        if duration <= 0:
            self.logger.warning("could not tell how long the audio is (usage: %s)", json.dumps(usage))
        self.logger.info("transcribed %.1fs of audio (usage: %s)", duration, json.dumps(usage))
        return AppOutput(
            text=data.get("text") or "",
            languages=[item.get("code") for item in data.get("languages") or [] if isinstance(item, dict) and item.get("code")],
            duration=round(duration, 3),
            output_meta=OutputMeta(inputs=[AudioMeta(seconds=round(duration, 3), extra={"model": FILE_MODEL})], outputs=[]),
        )

    async def realtime(self, input_data: RealtimeInput, socket: Socket) -> AsyncGenerator[RealtimeOutput, None]:
        """Transcribe a microphone while it speaks."""
        import websockets

        live = Live(socket, input_data, RealtimeOutput)
        self._socket = socket
        transcript = _Transcript()
        ready = asyncio.Event()                 # OpenAI accepted the session
        flushed = asyncio.Event()               # every committed turn has its final text
        snapshots: "asyncio.Queue[Any]" = asyncio.Queue()
        state: Dict[str, Any] = {
            "sent": 0,                          # bytes of audio sent to OpenAI
            "turn_at": 0,                       # where the turn in progress began, in bytes sent
            "speech": False,                    # somebody spoke in the turn in progress
            "loud_at": time.monotonic(),        # when the caller's audio was last loud enough to be speech
            "spoke_at": time.monotonic(),       # when OpenAI last heard words
            "open_turns": 0,                    # committed turns still waiting for their final text
            "caller_gone": False,
        }
        openai_error: Dict[str, str] = {}

        openai = await websockets.connect(
            REALTIME_URL,
            additional_headers={"Authorization": f"Bearer {self.api_key}"},
            max_size=None,
        )
        await openai.send(json.dumps({"type": "session.update", "session": _session(input_data)}))

        async def push() -> None:
            if not socket.closed:                 # the result is yielded after the caller has gone
                await live.send(text=transcript.text)

        async def commit() -> None:
            """Ends the turn in progress: OpenAI answers with its final text."""
            transcript.committed(state["turn_at"] / BYTES_PER_SECOND, state["sent"] / BYTES_PER_SECOND)
            state["turn_at"], state["speech"] = state["sent"], False
            state["open_turns"] += 1
            await openai.send(json.dumps({"type": "input_audio_buffer.commit"}))

        async def uplink() -> None:
            await ready.wait()
            pending = bytearray()
            async for update in live:
                if update.field in SESSION_FIELDS:
                    await openai.send(json.dumps({"type": "session.update", "session": _session(input_data)}))
                if update.field != "audio":
                    continue
                if _rms(update.value) >= SPEECH_RMS:
                    state["speech"], state["loud_at"] = True, time.monotonic()
                pending += update.value
                if len(pending) >= CHUNK_BYTES:
                    state["sent"] += len(pending)
                    await openai.send(json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(bytes(pending)).decode(),
                    }))
                    pending = bytearray()
            # The caller closed. Send what is left, commit the turn in
            # progress and give OpenAI a moment to answer it.
            state["caller_gone"] = True
            try:
                if pending:
                    state["sent"] += len(pending)
                    await openai.send(json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(bytes(pending)).decode(),
                    }))
                if state["speech"] and state["sent"] - state["turn_at"] >= MIN_TURN_BYTES:
                    await commit()
                if state["open_turns"] > 0:
                    await asyncio.wait_for(flushed.wait(), FLUSH_SECONDS)
            except Exception:  # noqa: BLE001 - OpenAI is gone or slow; what was finished is the result
                pass
            snapshots.put_nowait(_DONE)

        async def turn_watch() -> None:
            # gpt-live-transcribe finds no turns itself. A turn ends when the
            # caller's audio has been quiet (or absent: a gated microphone
            # sends nothing in a pause) for silence_ms after speech.
            await ready.wait()
            while not state["caller_gone"]:
                await asyncio.sleep(0.05)
                length = state["sent"] - state["turn_at"]
                if state["caller_gone"] or length < MIN_TURN_BYTES:
                    continue
                quiet = (time.monotonic() - state["loud_at"]) * 1000
                paused = state["speech"] and quiet >= input_data.silence_ms
                if paused or length >= MAX_TURN_SECONDS * BYTES_PER_SECOND:
                    await commit()

        async def downlink() -> None:
            try:
                async for message in openai:
                    if isinstance(message, bytes):
                        continue
                    event = json.loads(message)
                    kind = event.get("type", "")
                    if kind in SESSION_UPDATED:
                        if not ready.is_set():
                            ready.set()
                            await push()              # the first frame: the app is there
                    elif kind == "input_audio_buffer.committed":
                        transcript.claim(event.get("item_id"))
                    elif kind == DELTA:
                        if (event.get("delta") or "").strip():
                            state["spoke_at"] = time.monotonic()
                        if transcript.delta(event.get("item_id"), event.get("delta") or ""):
                            await push()
                    elif kind in (COMPLETED, FAILED):
                        if kind == FAILED:
                            self.logger.info("openai could not transcribe a turn: %s", json.dumps(event.get("error")))
                        changed = transcript.completed(event.get("item_id"), event.get("transcript") or "")
                        state["open_turns"] = max(0, state["open_turns"] - 1)
                        if changed:
                            await push()
                        snapshots.put_nowait(None)
                        if state["caller_gone"] and state["open_turns"] == 0:
                            flushed.set()
                    elif kind == "error":
                        # Most are recoverable (a rejected setting leaves the
                        # session up); OpenAI closes the socket when one is not.
                        err = event.get("error") or {}
                        openai_error["message"] = f"OpenAI: {err.get('message') or json.dumps(err)}"
                        self.logger.warning("%s", openai_error["message"])
                        if not ready.is_set():
                            snapshots.put_nowait(RuntimeError(openai_error["message"]))
                            return
                        await live.error(openai_error["message"])
                    elif kind not in QUIET_EVENTS:
                        self.logger.info("openai event %s: %s", kind, message[:300])
            except asyncio.CancelledError:
                raise
            except Exception as err:  # noqa: BLE001 - a closed socket ends the session, it does not fail it
                openai_error.setdefault("message", f"OpenAI closed the session: {err}")
            flushed.set()
            if state["caller_gone"]:
                return                              # the caller closed first; the result is on its way
            # OpenAI is gone. A session it never accepted is a failure; one it
            # accepted ends like any other and is billed.
            reason = openai_error.get("message") or "OpenAI closed the session"
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
                turns=transcript.turns(),
                seconds=round(state["sent"] / BYTES_PER_SECOND, 3),
                partial=partial,
            )

        end_reason = "the caller closed the session"
        tasks = [asyncio.create_task(job()) for job in (uplink, downlink, turn_watch, idle_watch)]
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
            await openai.close()

        # Whatever OpenAI heard of a turn it never finished is the last of the transcript.
        transcript.finish(state["sent"] / BYTES_PER_SECOND)
        seconds = state["sent"] / BYTES_PER_SECOND
        result = snapshot(partial=False)
        result.end_reason = end_reason
        # OpenAI bills a transcription session by the audio it was sent.
        result.output_meta = OutputMeta(
            inputs=[AudioMeta(seconds=round(seconds, 3), sample_rate=SAMPLE_RATE, extra={"model": LIVE_MODEL})],
            outputs=[],
        )
        self.logger.info("session ended (%s): %.1fs sent, %d turns", end_reason, seconds, len(result.turns))
        yield result


class _Transcript:
    """The transcript as OpenAI reports it: per item, words that add up as they
    are heard, then the final text once the item's turn is committed. Final
    texts can arrive out of order, so items keep the place they were first
    seen in."""

    def __init__(self) -> None:
        self._items: Dict[str, Dict[str, Any]] = {}     # item id -> {text, done, span}; in speech order
        self._spans: List[Tuple[float, float]] = []     # (start, end) of committed turns no item has claimed yet
        self._sent = ""

    @property
    def text(self) -> str:
        return " ".join(i["text"].strip() for i in self._items.values() if i["text"].strip())

    def _changed(self) -> bool:
        changed, self._sent = self.text != self._sent, self.text
        return changed

    def item(self, item_id: Optional[str]) -> Dict[str, Any]:
        """The item, placed where it is first seen."""
        key = item_id or ""
        if key not in self._items:
            self._items[key] = {"text": "", "done": False, "span": None}
        return self._items[key]

    def committed(self, start: float, end: float) -> None:
        """The app committed a turn covering this much of the stream."""
        self._spans.append((round(start, 3), round(end, 3)))

    def claim(self, item_id: Optional[str]) -> None:
        """OpenAI names the item of a committed turn; commits are answered in order."""
        item = self.item(item_id)
        if item["span"] is None and self._spans:
            item["span"] = self._spans.pop(0)

    def delta(self, item_id: Optional[str], text: str) -> bool:
        """Adds words to an item; True when the transcript changed."""
        item = self.item(item_id)
        if not item["done"]:
            item["text"] += text
        return self._changed()

    def completed(self, item_id: Optional[str], text: str) -> bool:
        """Sets an item's final text (empty when OpenAI could not transcribe it); True when the transcript changed."""
        item = self.item(item_id)
        item["text"], item["done"] = text or item["text"], True
        self.claim(item_id)
        return self._changed()

    def finish(self, now: float) -> None:
        """Closes items still open: what was heard of them is their text."""
        for item in self._items.values():
            if not item["done"]:
                item["done"] = True
                item["span"] = item["span"] or (self._spans.pop(0) if self._spans else (now, now))

    def turns(self) -> List[Turn]:
        found = []
        for item in self._items.values():
            if item["done"] and item["text"].strip():
                start, end = item["span"] or (0.0, 0.0)
                found.append(Turn(text=item["text"].strip(), start=start, end=end))
        return found


def _rms(pcm: bytes) -> float:
    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) // 2 * 2])
    if sys.byteorder == "big":
        samples.byteswap()
    return math.sqrt(sum(s * s for s in samples) / len(samples)) if samples else 0.0


def _audio_seconds(path: str) -> float:
    """Length of an audio file, 0 when it cannot be read."""
    try:
        import mutagen

        info = getattr(mutagen.File(path), "info", None)
        return float(getattr(info, "length", 0) or 0)
    except Exception:  # noqa: BLE001 - an unreadable file is measured as zero and logged by the caller
        return 0.0


def _session(input_data: RealtimeInput) -> Dict[str, Any]:
    transcription: Dict[str, Any] = {"model": LIVE_MODEL}
    if input_data.prompt:
        transcription["prompt"] = input_data.prompt
    if input_data.languages:
        transcription["languages"] = list(input_data.languages)
    if input_data.keyterms:
        transcription["keywords"] = list(input_data.keyterms)
    if input_data.delay:
        transcription["delay"] = input_data.delay
    return {
        "type": "transcription",
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                "transcription": transcription,
                "turn_detection": None,
            }
        },
    }

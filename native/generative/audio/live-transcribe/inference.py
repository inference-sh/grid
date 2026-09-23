"""Live dictation: microphone audio in, a transcript that refines as you speak.

The caller streams mic frames and gets the transcript back while it talks, the
way system dictation does: the tail of the sentence keeps being rewritten until
a pause settles it, and settled phrases never move again.

What the socket carries is declared by the models, the same way the request
body is: ``audio`` is a live field, and the ordinary fields can be changed
while the stream runs.

    caller -> app   <binary PCM s16le mono 16 kHz>   an item of TranscribeInput.audio (a frame every ~20 ms)
    caller -> app   {"language": "en"}                change an ordinary input mid-stream
    app -> caller   {"text": ""}                      the first frame, so the caller knows the app is there
    app -> caller   {"text": "hello there"}           TranscribeOutput.text, pushed on every refresh
    caller closes  ->  the last window is flushed and the function yields its result

The socket is the low-latency channel; the yields are the task's output. Each
yield is a cumulative snapshot (``partial=True``) that reaches the caller as a
task update over SSE, and the yield after the socket closes is the result and
carries ``output_meta``.
"""

import asyncio
import logging
import time
from typing import AsyncGenerator, List, Optional

import numpy as np
from pydantic import BaseModel, Field

from inferencesh import (
    AudioMeta,
    BaseApp,
    BaseAppInput,
    BaseAppOutput,
    BaseAppSetup,
    Live,
    OutputMeta,
    PCM16,
    Socket,
    Stream,
)

SAMPLE_RATE = 16000
REFRESH_SAMPLES = SAMPLE_RATE // 2        # new audio that earns another pass over the window
VAD_SAMPLES = SAMPLE_RATE // 10           # new audio that earns another look for the end of speech
MAX_WINDOW_SAMPLES = SAMPLE_RATE * 30     # whisper hears 30 s; commit before the window overflows
MIN_COMMIT_SAMPLES = SAMPLE_RATE // 4     # a window shorter than this is noise, not a phrase


class AppSetup(BaseAppSetup):
    model: str = Field(
        default="large-v3-turbo",
        description="faster-whisper model: a size (tiny, base, small, medium, large-v3, "
        "large-v3-turbo) or a CTranslate2 model id on the Hub. Smaller is faster to the "
        "first word; on a machine without a GPU only the small ones keep up with speech.",
    )


class TranscribeInput(BaseAppInput):
    audio: Stream[PCM16(SAMPLE_RATE)] = Field(description="Microphone audio, a frame every 20 ms or so")
    language: Optional[str] = Field(
        default=None, description="ISO language code such as 'en'. Left empty the language is detected."
    )
    initial_prompt: Optional[str] = Field(
        default=None, description="Vocabulary hint: names and terms the transcript should prefer"
    )
    silence_ms: int = Field(
        default=700, ge=200, le=5000, description="Trailing silence that settles a phrase"
    )


class Segment(BaseModel):
    """A settled phrase. Its times are seconds from the start of the stream."""

    text: str = Field(description="What was said")
    start: float = Field(description="Seconds from the start of the stream")
    end: float = Field(description="Seconds from the start of the stream")


class TranscribeOutput(BaseAppOutput):
    text: str = Field(default="", description="The transcript so far: settled phrases plus the live tail")
    partial: bool = Field(default=True, description="True while the tail may still change; False on the result")
    segments: List[Segment] = Field(default_factory=list, description="The settled phrases, in order")
    seconds: float = Field(default=0, description="Audio received, in seconds")


class App(BaseApp):
    async def setup(self, config: AppSetup, metadata):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self._cancel = False

        from faster_whisper import WhisperModel
        import ctranslate2

        # ctranslate2 is the runtime that would actually use the GPU, so ask it
        # rather than torch (which this app does not install anyway).
        cuda = ctranslate2.get_cuda_device_count() > 0
        device, compute_type = ("cuda", "float16") if cuda else ("cpu", "int8")

        started = time.monotonic()
        self.model = WhisperModel(config.model, device=device, compute_type=compute_type)
        self.logger.info(
            "loaded %s on %s/%s in %.1fs", config.model, device, compute_type, time.monotonic() - started
        )

        # Warm the Silero VAD that faster-whisper ships, so the first phrase is
        # not delayed by an ONNX session opening on the event loop.
        from faster_whisper.vad import get_vad_model

        get_vad_model()

    async def on_cancel(self):
        self._cancel = True
        return True

    async def transcribe(
        self, input_data: TranscribeInput, socket: Socket
    ) -> AsyncGenerator[TranscribeOutput, None]:
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        live = Live(socket, input_data, TranscribeOutput)
        self._cancel = False

        # Per-stream state: locals, because the worker is reused across callers.
        window = bytearray()        # PCM since the last commit
        window_start = 0.0          # where the window begins, seconds from the start of the stream
        total = 0                   # samples received
        settled: List[Segment] = []
        partial = ""                # the live tail: the window as last transcribed
        language = input_data.language  # pinned once detected, so the tail stops flipping languages
        job: Optional[asyncio.Task] = None
        job_end = 0                 # window length the in-flight pass covers
        done_at = 0                 # window length the last finished pass covered
        vad_at = 0                  # window length the last VAD look covered
        speech_end = 0              # last sample of the window that VAD called speech
        heard_speech = False

        def text_now() -> str:
            return " ".join([s.text for s in settled] + ([partial] if partial else [])).strip()

        async def push(is_partial: bool = True) -> TranscribeOutput:
            """The snapshot to yield, pushed over the socket on the way out."""
            body = text_now()
            if not socket.closed:                     # the result is yielded after the caller has gone
                await live.send(text=body)
            return TranscribeOutput(
                text=body, partial=is_partial, segments=list(settled), seconds=round(total / SAMPLE_RATE, 3)
            )

        # Speak first, so the caller knows the app is on the other end.
        await live.send(text="")

        async for update in live:
            if self._cancel:
                break
            if update.field != "audio":
                continue                              # an ordinary field; Live already applied it
            window += update.value
            total += len(update.value) // 2
            length = len(window) // 2

            # A finished pass is the new tail. Frames arrive every 20 ms, so this
            # picks the result up about that fast without ever awaiting it.
            if job is not None and job.done():
                partial, _, _, detected = job.result()
                language = language or detected
                done_at, job = job_end, None
                yield await push()

            # Look for the end of speech in the tail only: everything before it
            # was already silent or already speech. A tenth of a second of audio
            # per look keeps Silero's cost in the noise.
            if length - vad_at >= VAD_SAMPLES:
                vad_at = length
                tail_samples = min(length, int(SAMPLE_RATE * (input_data.silence_ms + 200) / 1000))
                tail = np.frombuffer(memoryview(window)[-tail_samples * 2 :], dtype="<i2").astype(np.float32) / 32768.0
                spans = get_speech_timestamps(
                    tail,
                    VadOptions(min_speech_duration_ms=100, min_silence_duration_ms=100, speech_pad_ms=0),
                    sampling_rate=SAMPLE_RATE,
                )
                if spans:
                    heard_speech = True
                    speech_end = length - tail_samples + spans[-1]["end"]

            silent_for = (length - speech_end) / SAMPLE_RATE * 1000
            if length >= MAX_WINDOW_SAMPLES or (heard_speech and silent_for >= input_data.silence_ms):
                # The one place worth waiting on a transcription: the window
                # cannot be cleared until its text is final. The caller keeps
                # sending and the kernel queues the frames, and what it is
                # sending right now is the silence that triggered this commit.
                if job is not None:
                    await job
                    job = None
                if heard_speech and length >= MIN_COMMIT_SAMPLES:
                    text, start, end, detected = await asyncio.to_thread(
                        self._pass, bytes(window), language, input_data.initial_prompt
                    )
                    language = language or detected
                    if text:
                        settled.append(
                            Segment(
                                text=text,
                                start=round(window_start + start, 3),
                                end=round(window_start + min(end, length / SAMPLE_RATE), 3),
                            )
                        )
                window_start += length / SAMPLE_RATE
                window.clear()
                partial, heard_speech = "", False
                speech_end = done_at = job_end = vad_at = 0
                yield await push()
                continue

            # One pass in flight at a time; it starts again once enough new
            # audio has arrived to be worth re-reading the window.
            if job is None and heard_speech and length - done_at >= REFRESH_SAMPLES:
                job_end = length
                job = asyncio.create_task(
                    asyncio.to_thread(self._pass, bytes(window), language, input_data.initial_prompt)
                )

        # The caller is gone. Finish what is in flight, then read the last
        # window so its words are not lost, and report.
        if job is not None:
            await asyncio.gather(job, return_exceptions=True)
            job = None
        length = len(window) // 2
        if heard_speech and length >= MIN_COMMIT_SAMPLES:
            text, start, end, _ = await asyncio.to_thread(
                self._pass, bytes(window), language, input_data.initial_prompt
            )
            if text:
                settled.append(
                    Segment(
                        text=text,
                        start=round(window_start + start, 3),
                        end=round(window_start + min(end, length / SAMPLE_RATE), 3),
                    )
                )
        partial = ""

        seconds = total / SAMPLE_RATE
        result = await push(is_partial=False)
        result.output_meta = OutputMeta(inputs=[AudioMeta(seconds=seconds)], outputs=[])
        self.logger.info("stream ended: %.1fs of audio, %d phrases", seconds, len(settled))
        yield result

    def _pass(self, pcm: bytes, language: Optional[str], prompt: Optional[str]):
        """One read of the whole window, in a worker thread.

        VAD already ran over this audio and the window is one phrase, so the
        decoder gets neither its own VAD nor a previous-text condition to drift
        on, and a beam of one keeps it inside the refresh interval.
        """
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        segments, info = self.model.transcribe(
            audio,
            language=language,
            initial_prompt=prompt,
            beam_size=1,
            word_timestamps=False,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        parts, start, end = [], None, 0.0
        for segment in segments:
            parts.append(segment.text.strip())
            start = segment.start if start is None else start
            end = segment.end
        return " ".join(p for p in parts if p).strip(), start or 0.0, end, info.language

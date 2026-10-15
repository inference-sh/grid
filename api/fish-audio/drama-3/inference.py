"""Fish Audio Drama 3: expressive text to speech on Fish Audio's API (`POST /v1/tts`, model `drama-3`).

Drama 3 is directed in the text itself: bracketed directions and sounds
(`[barely holding it together]`, `[small laugh]`), paired delivery tags
(`<whisper>…</whisper>`), language tags (`<french>…</french>`) and speaker
tags (`<|speaker:0|>`) for dialogue between several voices.

A voice is a Fish Audio voice id, or a reference clip with its transcript for
zero-shot cloning. Inline references must be sent as MessagePack, so every
request is.

Fish Audio bills per million UTF-8 bytes of input text; the output reports
them in `output_meta`.
"""

import asyncio
import logging
import os
import tempfile
from typing import List, Literal, Optional

import httpx
import msgpack
from inferencesh import AudioMeta, BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, function
from pydantic import Field, model_validator

BASE_URL = "https://api.fish.audio"
API_KEY_SECRET = "FISH_KEY"
MODEL = "drama-3"

# Fish Audio answers 429 without Retry-After when the account's concurrency is used up.
MAX_RETRIES = 3
RETRY_STATUSES = {429, 500, 502, 503, 504}

# Fish Audio's own per-generation limit on its Pro plan.
MAX_TEXT_CHARS = 30000
# Fish Audio recommends 10 to 30 seconds of clear speech; WAV, MP3 or FLAC.
MAX_REFERENCE_BYTES = 20 * 1024 * 1024

AudioFormat = Literal["mp3", "wav", "opus", "pcm"]
EXTENSIONS = {"mp3": ".mp3", "wav": ".wav", "opus": ".opus", "pcm": ".pcm"}
DEFAULT_SAMPLE_RATE = {"mp3": 44100, "wav": 44100, "pcm": 44100, "opus": 48000}


class UpstreamError(RuntimeError):
    """Fish Audio answered with an error. The message carries its status and body."""


class AppInput(BaseAppInput):
    text: str = Field(
        min_length=1,
        max_length=MAX_TEXT_CHARS,
        description=(
            "What to say, and how. Bracketed directions and sounds are performed, not read: "
            "[barely holding it together] I'm fine. [small laugh] Paired tags shape specific words: "
            "<whisper>, <emphasis>, <soft>, <fast>, <slow>, <stress>, <forceful>. "
            "For dialogue, mark turns with <|speaker:0|>, <|speaker:1|>, ... matching the order of voice_ids."
        ),
    )
    voice_ids: List[str] = Field(
        default_factory=list,
        description=(
            "Fish Audio voice ids (from fish.audio). One id for a single voice; several for a dialogue, "
            "one per speaker tag in order. Empty: the model picks a voice."
        ),
        examples=[["bf322df2096a46f18c579d0baa36f41d"]],
    )
    reference_audio: Optional[File] = Field(
        default=None,
        description="A clip of the voice to clone: 10 to 30 seconds of clear speech, WAV, MP3 or FLAC. Use it instead of voice_ids.",
    )
    reference_text: Optional[str] = Field(
        default=None,
        description="Exact transcript of reference_audio. Required with it.",
    )
    format: AudioFormat = Field(default="mp3", description="Audio format. pcm is raw 16-bit mono.")
    sample_rate: Optional[Literal[8000, 16000, 24000, 32000, 44100, 48000]] = Field(
        default=None,
        description="Sample rate in Hz. Empty: 44100, or 48000 for opus. mp3 takes 32000 or 44100; opus only 48000.",
    )
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speaking speed, 0.5 to 2.")
    temperature: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Sampling temperature, 0 to 1. Higher is more varied. Empty: Fish Audio's default."
    )
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Nucleus sampling, 0 to 1. Empty: Fish Audio's default.")
    normalize: bool = Field(default=True, description="Normalize numbers and symbols in English and Chinese text before speaking.")
    latency: Literal["normal", "balanced", "low"] = Field(
        default="normal", description="normal for best quality; balanced and low trade quality for a faster start."
    )

    @model_validator(mode="after")
    def check_voice(self):
        if self.reference_audio is not None and not (self.reference_text or "").strip():
            raise ValueError("reference_text is required with reference_audio")
        if self.reference_text and self.reference_audio is None:
            raise ValueError("reference_text needs reference_audio")
        if self.reference_audio is not None and self.voice_ids:
            raise ValueError("use either voice_ids or reference_audio, not both")
        if self.format == "opus" and self.sample_rate not in (None, 48000):
            raise ValueError("opus is 48000 Hz only")
        if self.format == "mp3" and self.sample_rate not in (None, 32000, 44100):
            raise ValueError("mp3 takes 32000 or 44100 Hz")
        if self.format != "mp3" and self.format != "opus" and self.sample_rate == 48000:
            raise ValueError("wav and pcm take 8000 to 44100 Hz")
        return self


class AppOutput(BaseAppOutput):
    audio: File = Field(description="The generated speech.")


def audio_seconds(path: str, fmt: str, sample_rate: int) -> float:
    """Length of the speech: from the size for raw PCM and WAV, from the headers otherwise."""
    if fmt == "pcm":
        return os.path.getsize(path) / (sample_rate * 2)
    try:
        import mutagen

        info = mutagen.File(path)
        if info is not None and info.info is not None:
            return float(info.info.length)
    except Exception:
        pass
    if fmt == "wav":
        return max(os.path.getsize(path) - 44, 0) / (sample_rate * 2)
    return 0.0


class App(BaseApp):
    async def setup(self, config):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        api_key = os.environ.get(API_KEY_SECRET)
        if not api_key:
            raise ValueError(f"{API_KEY_SECRET} is not set: add it as a secret")
        self.client = httpx.AsyncClient(
            base_url=BASE_URL,
            headers={"Authorization": f"Bearer {api_key}", "model": MODEL, "content-type": "application/msgpack"},
            timeout=httpx.Timeout(300, connect=15),
        )

    def _body(self, input_data: AppInput) -> dict:
        body = {
            "text": input_data.text,
            "format": input_data.format,
            "normalize": input_data.normalize,
            "latency": input_data.latency,
            "prosody": {"speed": input_data.speed},
        }
        if input_data.sample_rate:
            body["sample_rate"] = input_data.sample_rate
        if input_data.temperature is not None:
            body["temperature"] = input_data.temperature
        if input_data.top_p is not None:
            body["top_p"] = input_data.top_p
        if len(input_data.voice_ids) == 1:
            body["reference_id"] = input_data.voice_ids[0]
        elif input_data.voice_ids:
            body["reference_id"] = input_data.voice_ids
        if input_data.reference_audio is not None:
            path = input_data.reference_audio.path
            size = os.path.getsize(path)
            if size > MAX_REFERENCE_BYTES:
                raise ValueError("reference_audio is %d MB; the limit is %d MB" % (size // 2**20, MAX_REFERENCE_BYTES // 2**20))
            with open(path, "rb") as f:
                body["references"] = [{"audio": f.read(), "text": input_data.reference_text.strip()}]
        return body

    async def _speak(self, payload: bytes, out_path: str) -> None:
        """One request, streamed to a file: retried on 429 and 5xx, raises on any other error."""
        for attempt in range(MAX_RETRIES + 1):
            async with self.client.stream("POST", "/v1/tts", content=payload) as response:
                if response.status_code in RETRY_STATUSES and attempt < MAX_RETRIES:
                    delay = float(response.headers.get("Retry-After", 2 ** (attempt + 1)))
                    self.logger.warning("tts: %d, retrying in %.0fs", response.status_code, delay)
                    await asyncio.sleep(delay)
                    continue
                if response.is_error:
                    body = (await response.aread()).decode("utf-8", "replace")
                    raise UpstreamError("Fish Audio %d: %s" % (response.status_code, body[:500]))
                with open(out_path, "wb") as f:
                    async for chunk in response.aiter_bytes():
                        f.write(chunk)
                return

    @function(annotations={"title": "Text to speech", "readOnlyHint": False, "destructiveHint": False, "openWorldHint": True})
    async def run(self, input_data: AppInput) -> AppOutput:
        """Perform a script as speech, directed by the tags in the text."""
        body = self._body(input_data)
        payload = msgpack.packb(body, use_bin_type=True)
        text_bytes = len(input_data.text.encode("utf-8"))
        voice = "clone" if "references" in body else (",".join(input_data.voice_ids) or "model's choice")
        self.logger.info("tts: %d chars (%d bytes), voice %s, format %s", len(input_data.text), text_bytes, voice, input_data.format)

        fd, out_path = tempfile.mkstemp(suffix=EXTENSIONS[input_data.format])
        os.close(fd)
        await self._speak(payload, out_path)
        if os.path.getsize(out_path) == 0:
            raise UpstreamError("Fish Audio returned no audio")

        sample_rate = input_data.sample_rate or DEFAULT_SAMPLE_RATE[input_data.format]
        seconds = await asyncio.to_thread(audio_seconds, out_path, input_data.format, sample_rate)
        self.logger.info("tts: %.2fs of audio, %d bytes; usage: utf8_bytes=%d model=%s", seconds, os.path.getsize(out_path), text_bytes, MODEL)
        meta = AudioMeta(
            seconds=round(seconds, 3),
            sample_rate=sample_rate,
            extra={"utf8_bytes": text_bytes, "characters": len(input_data.text), "model": MODEL},
        )
        return AppOutput(audio=File(path=out_path), output_meta=OutputMeta(inputs=[], outputs=[meta]))

    async def unload(self):
        await self.client.aclose()

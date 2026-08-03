import logging
import os
from typing import Optional
from enum import Enum

import httpx
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, AudioMeta, TextMeta
from pydantic import Field

from .minimax_helper import download_file

TTS_URL = "https://api.minimax.io/v1/t2a_v2"
MODEL = "speech-2.8-hd"


class EmotionEnum(str, Enum):
    happy = "happy"
    sad = "sad"
    angry = "angry"
    fearful = "fearful"
    disgusted = "disgusted"
    surprised = "surprised"
    calm = "calm"
    fluent = "fluent"
    whisper = "whisper"


class FormatEnum(str, Enum):
    mp3 = "mp3"
    wav = "wav"
    flac = "flac"


class AppInput(BaseAppInput):
    """MiniMax Speech 2.8 HD — high-quality text-to-speech. 40 languages, 9 emotions."""

    text: str = Field(
        description="Text to convert to speech. Max 10,000 characters.",
        examples=["Hello, welcome to our platform. How can I help you today?"],
    )
    voice_id: str = Field(
        default="Wise_Woman",
        description="Voice ID — system voice name or custom cloned voice ID.",
    )
    speed: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Speech speed (0.5-2.0).",
    )
    emotion: Optional[EmotionEnum] = Field(
        default=None,
        description="Emotion style for the speech.",
    )
    format: FormatEnum = Field(
        default=FormatEnum.mp3,
        description="Output audio format.",
    )
    language_boost: Optional[str] = Field(
        default="auto",
        description="Language boost — 'auto' for auto-detect, or specify language code.",
    )


class AppOutput(BaseAppOutput):
    audio: File = Field(description="The generated audio file.")


class App(BaseApp):

    async def setup(self, metadata):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.api_key = os.environ.get("MINIMAX_KEY")
        if not self.api_key:
            raise RuntimeError(
                "MINIMAX_KEY is not set. Check that `belt secrets get MINIMAX_KEY "
                "--json` reports a non-empty masked_value."
            )
        self.client = httpx.AsyncClient(timeout=120)
        self.logger.info(f"{MODEL} initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        self.logger.info(f"Generating speech: {len(input_data.text)} chars, voice={input_data.voice_id}")

        voice_setting = {
            "voice_id": input_data.voice_id,
            "speed": input_data.speed,
        }
        if input_data.emotion:
            voice_setting["emotion"] = input_data.emotion.value

        payload = {
            "model": MODEL,
            "text": input_data.text,
            "voice_setting": voice_setting,
            "output_format": "url",
            "audio_setting": {
                "format": input_data.format.value,
                "sample_rate": 24000,
            },
        }
        if input_data.language_boost:
            payload["language_boost"] = input_data.language_boost

        resp = await self.client.post(
            TTS_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )

        if resp.status_code != 200:
            raise RuntimeError(f"MiniMax TTS error ({resp.status_code}): {resp.text[:300]}")

        data = resp.json()
        base_resp = data.get("base_resp") or {}
        if base_resp.get("status_code", 0) != 0:
            raise RuntimeError(f"MiniMax TTS error: {base_resp.get('status_msg', 'Unknown error')}")

        audio_url = (data.get("data") or {}).get("audio")
        if not audio_url:
            raise RuntimeError(f"No audio URL in response: {str(data)[:300]}")

        extra_info = data.get("extra_info") or {}
        audio_length_ms = float(extra_info.get("audio_length") or 0)
        usage_chars = int(extra_info.get("usage_characters") or len(input_data.text))

        self.logger.info(f"Audio ready: {audio_length_ms}ms, {usage_chars} chars")

        ext = input_data.format.value
        audio_path = await download_file(self.client, audio_url, f"/tmp/output.{ext}")

        audio_seconds = audio_length_ms / 1000.0 if audio_length_ms > 0 else 0

        output_meta = OutputMeta(
            inputs=[TextMeta(tokens=usage_chars, extra={"unit": "characters"})],
            outputs=[AudioMeta(
                seconds=audio_seconds,
                sample_rate=int(extra_info.get("audio_sample_rate") or 24000),
                extra={"model": MODEL, "voice_id": input_data.voice_id},
            )],
        )

        return AppOutput(audio=File(path=audio_path), output_meta=output_meta)

    async def on_cancel(self):
        self.logger.info("Cancellation requested")
        return True

    async def unload(self):
        await self.client.aclose()

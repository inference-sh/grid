"""
Grok TTS - xAI Text to Speech

Convert text into natural speech using xAI's Text to Speech API.
Supports multiple voices, expressive speech tags, and multiple audio formats.
"""

import os
import logging
import tempfile
from typing import Literal

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, AudioMeta
from pydantic import Field
import requests


VoiceType = Literal["eve", "ara", "rex", "sal", "leo"]

CodecType = Literal["mp3", "wav", "pcm", "mulaw", "alaw"]

LanguageType = Literal[
    "auto", "en",
    "ar-EG", "ar-SA", "ar-AE",
    "bn", "zh", "fr", "de", "hi", "id", "it", "ja", "ko",
    "pt-BR", "pt-PT", "ru", "es-MX", "es-ES", "tr", "vi",
]

CODEC_EXTENSIONS = {
    "mp3": ".mp3",
    "wav": ".wav",
    "pcm": ".pcm",
    "mulaw": ".ulaw",
    "alaw": ".alaw",
}

CODEC_CONTENT_TYPES = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
    "mulaw": "audio/basic",
    "alaw": "audio/basic",
}


class AppInput(BaseAppInput):
    """Input schema for xAI Text to Speech."""

    text: str = Field(
        description=(
            "Text to convert to speech (max 15,000 characters). "
            "Supports inline speech tags: [pause], [laugh], [sigh], etc. "
            "and wrapping tags: <soft>, <whisper>, <loud>, <singing>, etc."
        ),
        max_length=15000,
        examples=["Hello, this is a text-to-speech test from xAI."],
    )
    voice: VoiceType = Field(
        default="eve",
        description="Voice to use. Options: eve, ara, rex, sal, leo.",
    )
    language: LanguageType = Field(
        default="auto",
        description="BCP-47 language code or 'auto' for automatic detection.",
    )
    codec: CodecType = Field(
        default="mp3",
        description="Audio output format.",
    )
    sample_rate: Literal[8000, 16000, 22050, 24000, 44100, 48000] = Field(
        default=24000,
        description="Sample rate in Hz.",
    )


class AppOutput(BaseAppOutput):
    """Output schema for xAI Text to Speech."""

    audio: File = Field(description="The generated audio file.")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

XAI_TTS_URL = "https://api.x.ai/v1/tts"


class App(BaseApp):
    """xAI Text to Speech application."""

    async def setup(self):
        """Initialize the xAI API credentials."""
        self.api_key = os.environ.get("XAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("XAI_API_KEY environment variable is required")
        logger.info("Grok TTS initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        """Convert text to speech using the xAI TTS API."""
        logger.info(f"TTS request: voice={input_data.voice}, language={input_data.language}, "
                     f"codec={input_data.codec}, text_length={len(input_data.text)}")

        payload = {
            "text": input_data.text,
            "voice_id": input_data.voice,
            "language": input_data.language,
            "response_format": input_data.codec,
            "sample_rate": input_data.sample_rate,
        }

        response = requests.post(
            XAI_TTS_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=120,
        )

        if not response.ok:
            error_text = response.text[:500]
            if response.status_code == 429:
                raise RuntimeError(f"xAI rate limit exceeded. Try again shortly. Detail: {error_text}")
            elif response.status_code == 401 or response.status_code == 403:
                raise RuntimeError("xAI authentication failed. The API key may be invalid or expired.")
            elif response.status_code == 400:
                raise RuntimeError(f"xAI rejected the request — check your text or parameters. Detail: {error_text}")
            else:
                raise RuntimeError(f"xAI TTS failed (HTTP {response.status_code}): {error_text}")

        ext = CODEC_EXTENSIONS.get(input_data.codec, ".mp3")
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
            f.write(response.content)
            audio_path = f.name

        content_type = CODEC_CONTENT_TYPES.get(input_data.codec, "audio/mpeg")

        output_meta = OutputMeta(
            outputs=[
                AudioMeta(
                    content_type=content_type,
                    extra={
                        "voice": input_data.voice,
                        "language": input_data.language,
                        "codec": input_data.codec,
                        "text_length": len(input_data.text),
                    },
                )
            ]
        )

        logger.info(f"TTS completed: {len(response.content)} bytes, format={input_data.codec}")

        return AppOutput(
            audio=File(path=audio_path),
            output_meta=output_meta,
        )

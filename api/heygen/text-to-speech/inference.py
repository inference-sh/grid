"""
HeyGen Text-to-Speech - High-quality TTS with the Starfish engine.

Generate natural speech audio from text with configurable voice,
speed, and language settings.
"""

import logging
from typing import Optional, Literal

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, AudioMeta
from pydantic import Field

from .heygen_helper import (
    get_client,
    post_endpoint,
    download_file,
)


InputType = Literal["text", "ssml"]


class AppInput(BaseAppInput):
    """Input for text-to-speech."""

    text: str = Field(
        description="Text to convert to speech (1-5000 characters).",
        examples=["Hello! Welcome to our platform. We're excited to have you here."],
    )
    voice_id: str = Field(
        description="Voice ID to use. Must support the Starfish engine.",
    )
    input_type: InputType = Field(
        default="text",
        description="Input format: 'text' for plain text, 'ssml' for SSML markup.",
    )
    speed: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Speech speed multiplier (0.5-2.0).",
    )
    language: Optional[str] = Field(
        default=None,
        description="ISO language code (e.g., 'en', 'es', 'pt').",
    )
    locale: Optional[str] = Field(
        default=None,
        description="BCP-47 locale (e.g., 'en-US', 'pt-BR').",
    )


class AppOutput(BaseAppOutput):
    """Output from text-to-speech."""

    audio: File = Field(description="The generated speech audio file.")


class App(BaseApp):
    async def setup(self):
        self.logger = logging.getLogger(__name__)

    async def run(self, input_data: AppInput) -> AppOutput:
        self.logger.info(f"Generating speech: {input_data.text[:80]}...")

        payload = {
            "text": input_data.text,
            "voice_id": input_data.voice_id,
            "input_type": input_data.input_type,
            "speed": input_data.speed,
        }

        if input_data.language:
            payload["language"] = input_data.language
        if input_data.locale:
            payload["locale"] = input_data.locale

        async with get_client() as client:
            result = await post_endpoint(client, "/v3/voices/speech", payload)

        audio_url = result.get("audio_url")
        if not audio_url:
            raise RuntimeError("No audio_url in TTS response")

        audio_path = await download_file(audio_url, suffix=".mp3")
        duration = result.get("duration", 0)

        output_meta = OutputMeta(
            outputs=[
                AudioMeta(
                    duration_seconds=float(duration),
                )
            ]
        )

        self.logger.info(f"TTS complete: {duration}s")
        return AppOutput(audio=File(path=audio_path), output_meta=output_meta)

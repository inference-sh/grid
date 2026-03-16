"""
ElevenLabs Dubbing

Automatically dub audio and video content across languages.
Supports 29 languages with automatic speaker detection.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, AudioMeta
from pydantic import Field
from typing import Optional, Literal
import logging

from .elevenlabs_helper import create_dubbing, get_api_key, get_audio_duration


# Supported target languages
TARGET_LANGUAGES = Literal[
    "en", "es", "fr", "de", "it", "pt", "pl", "hi", "ar", "zh",
    "ja", "ko", "ru", "tr", "nl", "sv", "da", "fi", "no", "cs",
    "el", "he", "hu", "id", "ms", "ro", "th", "uk", "vi",
]


class AppInput(BaseAppInput):
    """Input schema for ElevenLabs Dubbing."""

    audio: File = Field(
        description="Audio or video file to dub (MP3, MP4, WAV, MOV).",
    )
    target_lang: TARGET_LANGUAGES = Field(
        default="es",
        description="Target language code (e.g., 'es' for Spanish, 'fr' for French).",
    )
    source_lang: Optional[str] = Field(
        default=None,
        description="Source language code. Leave empty for auto-detection.",
    )


class AppOutput(BaseAppOutput):
    """Output schema for ElevenLabs Dubbing."""
    audio: File = Field(description="Dubbed audio file in target language")


class App(BaseApp):
    """ElevenLabs Dubbing app implementation."""

    async def setup(self):
        """Initialize the application."""
        self.logger = logging.getLogger(__name__)
        get_api_key()
        self.logger.info("ElevenLabs Dubbing app initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        """Dub audio to target language."""
        self.logger.info(f"Dubbing to: {input_data.target_lang}")

        audio_path = await create_dubbing(
            audio=input_data.audio.path,
            target_lang=input_data.target_lang,
            source_lang=input_data.source_lang,
            logger=self.logger,
        )

        # Get duration from output file for billing
        output_duration = get_audio_duration(audio_path, self.logger)

        return AppOutput(
            audio=File(path=audio_path),
            output_meta=OutputMeta(
                inputs=[AudioMeta(seconds=output_duration)],
                outputs=[AudioMeta(seconds=output_duration)]
            )
        )

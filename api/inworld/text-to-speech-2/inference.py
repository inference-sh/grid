"""
Inworld Text to Speech 2

High-quality multilingual TTS with 100+ languages.
Supports delivery modes (STABLE, BALANCED, CREATIVE) and
natural-language steering via square brackets.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, AudioMeta
from pydantic import Field
from typing import Optional, Literal
import logging

from .inworld_helper import text_to_speech, get_api_key, get_audio_duration


class AppInput(BaseAppInput):
    """Input schema for Inworld TTS-2."""

    text: str = Field(
        description="Text to convert to speech (max 2,000 characters). Use [brackets] for steering: emotion, volume, speed, non-verbals like [laugh] or [sigh].",
        max_length=2000,
    )
    voice_id: str = Field(
        description="Voice ID to use. List available voices via the Inworld platform.",
    )
    language: Optional[str] = Field(
        default=None,
        description="BCP-47 language tag (e.g. 'en-US', 'fr-FR', 'ja-JP'). Auto-detected if omitted. 100+ languages supported.",
    )
    delivery_mode: Literal[
        "STABLE",
        "BALANCED",
        "CREATIVE",
    ] = Field(
        default="BALANCED",
        description="Controls voice performance style. STABLE = consistent, BALANCED = natural, CREATIVE = expressive.",
    )
    audio_encoding: Literal[
        "MP3", "WAV", "OGG_OPUS", "FLAC", "LINEAR16", "ALAW", "MULAW",
    ] = Field(
        default="MP3",
        description="Audio output format.",
    )
    sample_rate_hertz: int = Field(
        default=44100,
        description="Audio sample rate in Hz. Supported: 8000, 16000, 22050, 24000, 32000, 44100, 48000.",
    )
    speaking_rate: float = Field(
        default=1.0,
        ge=0.5,
        le=1.5,
        description="Speaking rate multiplier (0.5-1.5).",
    )


class AppOutput(BaseAppOutput):
    """Output schema for Inworld TTS-2."""
    audio: File = Field(description="Generated speech audio file")


class App(BaseApp):
    """Inworld TTS-2 app implementation."""

    async def setup(self):
        self.logger = logging.getLogger(__name__)
        get_api_key()
        self.logger.info("Inworld TTS-2 app initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        self.logger.info(f"Generating speech: {len(input_data.text)} characters")
        self.logger.info(f"Voice: {input_data.voice_id}, Delivery: {input_data.delivery_mode}")

        audio_path = await text_to_speech(
            text=input_data.text,
            voice_id=input_data.voice_id,
            model_id="inworld-tts-2",
            audio_encoding=input_data.audio_encoding,
            sample_rate_hertz=input_data.sample_rate_hertz,
            speaking_rate=input_data.speaking_rate,
            delivery_mode=input_data.delivery_mode,
            language=input_data.language,
            logger=self.logger,
        )

        duration = get_audio_duration(audio_path, self.logger)

        return AppOutput(
            audio=File(path=audio_path),
            output_meta=OutputMeta(
                inputs=[],
                outputs=[AudioMeta(
                    seconds=duration,
                    extra={"characters": len(input_data.text), "model": "inworld-tts-2"},
                )],
            ),
        )

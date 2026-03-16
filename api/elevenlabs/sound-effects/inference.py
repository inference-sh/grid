"""
ElevenLabs Sound Effects

Generate custom sound effects from text descriptions.
Royalty-free audio output in MP3 format.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, TextMeta, AudioMeta
from pydantic import Field
from typing import Optional
import logging

from .elevenlabs_helper import generate_sound_effect, get_api_key, get_audio_duration


class AppInput(BaseAppInput):
    """Input schema for ElevenLabs Sound Effects."""

    text: str = Field(
        description="Description of the sound effect to generate (e.g., 'Cinematic Braam, Horror').",
        max_length=1000,
    )
    duration_seconds: Optional[float] = Field(
        default=None,
        ge=0.5,
        le=22.0,
        description="Duration in seconds (0.5-22). Leave empty for automatic duration.",
    )
    prompt_influence: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="How much the prompt influences generation (0-1). Higher = more literal interpretation.",
    )


class AppOutput(BaseAppOutput):
    """Output schema for ElevenLabs Sound Effects."""
    audio: File = Field(description="Generated sound effect audio file")


class App(BaseApp):
    """ElevenLabs Sound Effects app implementation."""

    async def setup(self):
        """Initialize the application."""
        self.logger = logging.getLogger(__name__)
        get_api_key()
        self.logger.info("ElevenLabs Sound Effects app initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        """Generate sound effect from text."""
        self.logger.info(f"Generating sound effect: {input_data.text[:50]}...")

        audio_path = generate_sound_effect(
            text=input_data.text,
            duration_seconds=input_data.duration_seconds,
            prompt_influence=input_data.prompt_influence,
            logger=self.logger,
        )

        # Get actual duration for output meta (priced per generation, but track duration)
        duration = get_audio_duration(audio_path, self.logger)

        return AppOutput(
            audio=File(path=audio_path),
            output_meta=OutputMeta(
                inputs=[TextMeta(text=input_data.text)],
                outputs=[AudioMeta(seconds=duration)]
            )
        )

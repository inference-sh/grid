"""
ElevenLabs Music (Eleven Music)

Generate studio-quality music from text prompts.
Create tracks up to 10 minutes with full commercial licensing.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, TextMeta, AudioMeta
from pydantic import Field
import logging

from .elevenlabs_helper import compose_music, get_api_key


class AppInput(BaseAppInput):
    """Input schema for ElevenLabs Music."""

    prompt: str = Field(
        description="Description of the music to generate (genre, mood, instruments, tempo, etc.).",
        max_length=2000,
    )
    duration_seconds: int = Field(
        default=30,
        ge=5,
        le=600,
        description="Duration in seconds (5-600, max 10 minutes).",
    )


class AppOutput(BaseAppOutput):
    """Output schema for ElevenLabs Music."""
    audio: File = Field(description="Generated music audio file")


class App(BaseApp):
    """ElevenLabs Music app implementation."""

    async def setup(self):
        """Initialize the application."""
        self.logger = logging.getLogger(__name__)
        get_api_key()
        self.logger.info("ElevenLabs Music app initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        """Generate music from text prompt."""
        self.logger.info(f"Composing music: {input_data.prompt[:50]}...")
        self.logger.info(f"Duration: {input_data.duration_seconds}s")

        audio_path = compose_music(
            prompt=input_data.prompt,
            duration_seconds=input_data.duration_seconds,
            logger=self.logger,
        )

        # Music is priced per minute, use requested duration
        return AppOutput(
            audio=File(path=audio_path),
            output_meta=OutputMeta(
                inputs=[TextMeta(text=input_data.prompt)],
                outputs=[AudioMeta(seconds=float(input_data.duration_seconds))]
            )
        )

"""
ElevenLabs Voice Isolator

Remove background noise, reverb, and interference from audio.
Isolates the human voice for clean recordings.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, AudioMeta
from pydantic import Field
import logging

from .elevenlabs_helper import isolate_voice, get_api_key, get_audio_duration


class AppInput(BaseAppInput):
    """Input schema for ElevenLabs Voice Isolator."""

    audio: File = Field(
        description="Audio file to process (WAV, MP3, FLAC, OGG, AAC). Max 500MB/1 hour.",
    )


class AppOutput(BaseAppOutput):
    """Output schema for ElevenLabs Voice Isolator."""
    audio: File = Field(description="Isolated voice audio with background noise removed")


class App(BaseApp):
    """ElevenLabs Voice Isolator app implementation."""

    async def setup(self):
        """Initialize the application."""
        self.logger = logging.getLogger(__name__)
        get_api_key()
        self.logger.info("ElevenLabs Voice Isolator app initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        """Isolate voice from background noise."""
        self.logger.info(f"Isolating voice from: {input_data.audio.path}")

        # Get input duration for billing (priced per minute)
        input_duration = get_audio_duration(input_data.audio.path, self.logger)

        audio_path = isolate_voice(
            audio_path=input_data.audio.path,
            logger=self.logger,
        )

        return AppOutput(
            audio=File(path=audio_path),
            output_meta=OutputMeta(
                inputs=[AudioMeta(seconds=input_duration)],
                outputs=[AudioMeta(seconds=input_duration)]
            )
        )

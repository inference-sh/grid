"""
ElevenLabs Forced Alignment

Align text to audio, generating precise word-level timestamps.
Useful for subtitles, karaoke, and audio editing.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, AudioMeta
from pydantic import Field
from typing import List, Optional
import logging

from .elevenlabs_helper import forced_alignment, get_api_key, get_audio_duration


class AppInput(BaseAppInput):
    """Input schema for ElevenLabs Forced Alignment."""

    audio: File = Field(
        description="Audio file to align text to.",
    )
    text: str = Field(
        description="Text that matches the audio content.",
    )


class AlignedWord(BaseAppInput):
    """Word with timing information."""
    text: str = Field(description="The word")
    start: float = Field(description="Start time in seconds")
    end: float = Field(description="End time in seconds")


class AppOutput(BaseAppOutput):
    """Output schema for ElevenLabs Forced Alignment."""
    words: List[dict] = Field(description="Word-level alignment with timestamps")
    text: str = Field(description="Full aligned text")


class App(BaseApp):
    """ElevenLabs Forced Alignment app implementation."""

    async def setup(self):
        """Initialize the application."""
        self.logger = logging.getLogger(__name__)
        get_api_key()
        self.logger.info("ElevenLabs Forced Alignment app initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        """Align text to audio."""
        self.logger.info(f"Aligning text to audio: {input_data.audio.path}")

        # Get audio duration for billing
        duration = get_audio_duration(input_data.audio.path, self.logger)

        result = forced_alignment(
            audio_path=input_data.audio.path,
            text=input_data.text,
            logger=self.logger,
        )

        # Get duration from alignment result if available
        words = result.get("words", [])
        if words and len(words) > 0 and duration == 0.0:
            duration = float(words[-1].get("end", 0))

        return AppOutput(
            words=words,
            text=result.get("text", input_data.text),
            output_meta=OutputMeta(
                inputs=[AudioMeta(seconds=duration)],
                outputs=[]
            )
        )

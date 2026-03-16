"""
ElevenLabs Text to Dialogue

Generate immersive, natural-sounding dialogue from text with multiple voices.
Each segment can have its own voice and performance directions.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, TextMeta, AudioMeta
from pydantic import Field, BaseModel
from typing import List, Literal
import logging

from .elevenlabs_helper import text_to_dialogue, get_api_key, get_audio_duration, get_voice_id

VoiceName = Literal[
    "adam", "alice", "aria", "bella", "bill", "brian", "callum", "charlie",
    "chris", "daniel", "eric", "george", "harry", "jessica", "laura", "liam",
    "lily", "matilda", "river", "roger", "sarah", "will",
]


class DialogueSegment(BaseModel):
    """A single dialogue segment with text and voice."""
    text: str = Field(description="Text to speak, can include directions like [cheerfully]")
    voice: VoiceName = Field(description="Voice for this segment")


class AppInput(BaseAppInput):
    """Input schema for ElevenLabs Text to Dialogue."""

    segments: List[DialogueSegment] = Field(
        description="List of dialogue segments, each with text and voice_id.",
        min_length=1,
    )


class AppOutput(BaseAppOutput):
    """Output schema for ElevenLabs Text to Dialogue."""
    audio: File = Field(description="Generated dialogue audio file")


class App(BaseApp):
    """ElevenLabs Text to Dialogue app implementation."""

    async def setup(self):
        """Initialize the application."""
        self.logger = logging.getLogger(__name__)
        get_api_key()
        self.logger.info("ElevenLabs Text to Dialogue app initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        """Generate dialogue from text segments."""
        self.logger.info(f"Generating dialogue with {len(input_data.segments)} segments")

        # Convert to API format
        inputs = [
            {"text": seg.text, "voice_id": get_voice_id(seg.voice)}
            for seg in input_data.segments
        ]

        # Calculate total character count for pricing
        total_chars = sum(len(seg.text) for seg in input_data.segments)

        audio_path = text_to_dialogue(
            inputs=inputs,
            logger=self.logger,
        )

        # Get actual duration
        duration = get_audio_duration(audio_path, self.logger)

        return AppOutput(
            audio=File(path=audio_path),
            output_meta=OutputMeta(
                inputs=[TextMeta(text="x" * total_chars)],  # Track character count
                outputs=[AudioMeta(seconds=duration)]
            )
        )

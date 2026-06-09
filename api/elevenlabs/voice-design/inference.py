"""
ElevenLabs Voice Design

Design a new AI voice from a text description.
Returns preview samples with voice IDs that can be used with TTS, voice changer,
and text-to-dialogue apps.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, AudioMeta
from pydantic import Field, BaseModel
from typing import Optional, List
import logging

from .elevenlabs_helper import design_voice, get_api_key


class VoicePreview(BaseModel):
    """A generated voice preview."""
    voice_id: str = Field(description="Voice ID to use with other ElevenLabs apps")
    audio: File = Field(description="Preview audio sample")
    duration_secs: float = Field(description="Duration of the preview in seconds")


class AppInput(BaseAppInput):
    """Input schema for ElevenLabs Voice Design."""

    description: str = Field(
        description="Describe the voice you want to create (e.g. 'A warm, mature British female voice with a storytelling quality and slight rasp').",
    )
    preview_text: Optional[str] = Field(
        default=None,
        description="Text to speak in the previews (min 100 characters). If not provided, text is auto-generated.",
        min_length=100,
        max_length=500,
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducible results.",
    )


class AppOutput(BaseAppOutput):
    """Output schema for ElevenLabs Voice Design."""
    previews: List[VoicePreview] = Field(description="Generated voice previews — pick a voice_id to use with TTS, voice changer, or dialogue apps")


class App(BaseApp):
    """ElevenLabs Voice Design app."""

    async def setup(self):
        self.logger = logging.getLogger(__name__)
        get_api_key()
        self.logger.info("ElevenLabs Voice Design app initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        self.logger.info(f"Designing voice: {input_data.description[:50]}")

        results = design_voice(
            voice_description=input_data.description,
            text=input_data.preview_text,
            seed=input_data.seed,
            logger=self.logger,
        )

        previews = [
            VoicePreview(
                voice_id=r["generated_voice_id"],
                audio=File(path=r["audio_path"]),
                duration_secs=r["duration_secs"],
            )
            for r in results
        ]

        return AppOutput(previews=previews)

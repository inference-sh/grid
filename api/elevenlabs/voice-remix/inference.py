"""
ElevenLabs Voice Remix

Remix an existing voice by changing its characteristics — accent, gender,
speaking style, pacing, or audio quality. Returns previews with new voice IDs.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, AudioMeta
from pydantic import Field, BaseModel
from typing import Optional, List
import logging

from .elevenlabs_helper import remix_voice, get_api_key


class VoicePreview(BaseModel):
    """A remixed voice preview."""
    voice_id: str = Field(description="Voice ID to use with other ElevenLabs apps")
    audio: File = Field(description="Preview audio sample")
    duration_secs: float = Field(description="Duration of the preview in seconds")


class AppInput(BaseAppInput):
    """Input schema for ElevenLabs Voice Remix."""

    voice_id: str = Field(
        description="Voice ID to remix (from voice-clone, voice-design, or a premade voice ID).",
    )
    description: str = Field(
        description="Describe how to modify the voice (e.g. 'Make it sound younger with an American accent and faster pacing').",
    )
    preview_text: Optional[str] = Field(
        default=None,
        description="Text to speak in the previews. If not provided, text is auto-generated.",
        max_length=500,
    )
    prompt_strength: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="How much to apply the remix description (0-1). Lower preserves more of the original voice.",
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducible results.",
    )


class AppOutput(BaseAppOutput):
    """Output schema for ElevenLabs Voice Remix."""
    previews: List[VoicePreview] = Field(description="Remixed voice previews — pick a voice_id to use with TTS, voice changer, or dialogue apps")


class App(BaseApp):
    """ElevenLabs Voice Remix app."""

    async def setup(self):
        self.logger = logging.getLogger(__name__)
        get_api_key()
        self.logger.info("ElevenLabs Voice Remix app initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        self.logger.info(f"Remixing voice {input_data.voice_id}: {input_data.description[:50]}")

        results = remix_voice(
            voice_id=input_data.voice_id,
            voice_description=input_data.description,
            text=input_data.preview_text,
            prompt_strength=input_data.prompt_strength,
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

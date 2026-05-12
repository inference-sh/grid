"""
Inworld Voice Design

Generate a custom voice from a text description.
Describe the voice you want (age, gender, accent, tone, pitch)
and get up to 3 preview samples. Publish the best one to use
with any Inworld TTS model (TTS-2, TTS 1.5 Max, TTS 1.5 Mini).
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from typing import Optional, List, Literal
import base64
import tempfile
import logging

from .inworld_helper import design_voice, publish_voice, get_api_key


class AppInput(BaseAppInput):
    """Input schema for voice design."""

    design_prompt: str = Field(
        description="Text description of the desired voice (30-250 characters, English). Include age, gender, accent, tone, pitch, and personality. Example: 'A warm, friendly mid-30s female voice with a slight British accent, conversational and upbeat tone.'",
        min_length=30,
        max_length=250,
    )
    preview_text: str = Field(
        description="Text the generated voice will speak in the preview (should result in 1-15 seconds of audio).",
    )
    language: Literal[
        "EN_US", "ZH_CN", "KO_KR", "JA_JP", "RU_RU", "AUTO",
        "IT_IT", "ES_ES", "PT_BR", "DE_DE", "FR_FR", "AR_SA",
        "PL_PL", "NL_NL", "HI_IN", "HE_IL",
    ] = Field(
        default="EN_US",
        description="Language for the designed voice.",
    )
    number_of_samples: int = Field(
        default=3,
        ge=1,
        le=3,
        description="Number of voice previews to generate (1-3). More samples = more options to choose from.",
    )


class VoicePreview(BaseAppOutput):
    """A single voice preview."""
    voice_id: str = Field(description="Temporary voice ID for this preview — use with the publish function to save it")
    audio: File = Field(description="Audio preview of this voice")


class AppOutput(BaseAppOutput):
    """Output schema for voice design."""
    previews: List[VoicePreview] = Field(description="Generated voice previews — listen and pick the best one, then publish it")
    total: int = Field(description="Number of previews generated")


class PublishInput(BaseAppInput):
    """Input for publishing a designed voice."""
    voice_id: str = Field(
        description="Voice ID from a design preview to publish permanently.",
    )
    display_name: str = Field(
        description="Name for the published voice (e.g. 'Friendly Narrator').",
    )
    description: Optional[str] = Field(
        default=None,
        description="Optional description of the voice.",
    )


class PublishOutput(BaseAppOutput):
    """Output for published voice."""
    voice_id: str = Field(description="Permanent voice ID — use this with any Inworld TTS model")
    display_name: str = Field(description="Display name of the published voice")


class App(BaseApp):
    """Inworld Voice Design app."""

    async def setup(self):
        self.logger = logging.getLogger(__name__)
        get_api_key()
        self.logger.info("Inworld Voice Design app initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        """Design a voice from a text description. Returns preview samples."""
        self.logger.info(f"Designing voice: {input_data.design_prompt[:80]}")

        result = await design_voice(
            design_prompt=input_data.design_prompt,
            preview_text=input_data.preview_text,
            lang_code=input_data.language,
            number_of_samples=input_data.number_of_samples,
            logger=self.logger,
        )

        previews = []
        for i, pv in enumerate(result.get("previewVoices", [])):
            audio_b64 = pv.get("previewAudio", "")
            if not audio_b64:
                continue

            audio_bytes = base64.b64decode(audio_b64)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp.write(audio_bytes)
                audio_path = tmp.name

            self.logger.info(f"Preview {i+1}: voice_id={pv.get('voiceId')}, {len(audio_bytes)} bytes")

            previews.append(VoicePreview(
                voice_id=pv.get("voiceId", ""),
                audio=File(path=audio_path),
            ))

        return AppOutput(previews=previews, total=len(previews))

    async def publish(self, input_data: PublishInput) -> PublishOutput:
        """Publish a designed voice preview to make it permanently available."""
        self.logger.info(f"Publishing voice: {input_data.voice_id} as {input_data.display_name}")

        result = await publish_voice(
            voice_id=input_data.voice_id,
            display_name=input_data.display_name,
            description=input_data.description,
            logger=self.logger,
        )

        return PublishOutput(
            voice_id=result.get("voiceId", input_data.voice_id),
            display_name=result.get("displayName", input_data.display_name),
        )

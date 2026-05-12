"""
Inworld Voice Cloning

Instantly clone a voice from 5-15 seconds of audio.
The cloned voice ID can be used with any Inworld TTS model
(TTS-2, TTS 1.5 Max, TTS 1.5 Mini).
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from typing import Optional, Literal
import logging

from .inworld_helper import clone_voice, text_to_speech, get_api_key, get_audio_duration


class AppInput(BaseAppInput):
    """Input schema for voice cloning."""

    audio: File = Field(
        description="Audio sample of the voice to clone (WAV or MP3, 5-15 seconds). Longer clips are cut off at 15s.",
    )
    display_name: str = Field(
        description="Name for the cloned voice (e.g. 'My Custom Voice').",
    )
    language: Literal[
        "EN_US", "ZH_CN", "KO_KR", "JA_JP", "RU_RU", "AUTO",
        "IT_IT", "ES_ES", "PT_BR", "DE_DE", "FR_FR", "AR_SA",
        "PL_PL", "NL_NL", "HI_IN", "HE_IL",
    ] = Field(
        default="EN_US",
        description="Language of the voice sample.",
    )
    description: Optional[str] = Field(
        default=None,
        description="Optional description of the voice (e.g. 'Warm female narrator with British accent').",
    )
    remove_background_noise: bool = Field(
        default=True,
        description="Remove background noise from the audio sample before cloning.",
    )
    preview_text: Optional[str] = Field(
        default=None,
        description="Optional text to generate a preview with the cloned voice. If provided, returns a preview audio file.",
        max_length=2000,
    )


class AppOutput(BaseAppOutput):
    """Output schema for voice cloning."""

    voice_id: str = Field(description="The cloned voice ID — use this with any Inworld TTS model")
    display_name: str = Field(description="Display name of the cloned voice")
    language: str = Field(description="Language code of the cloned voice")
    preview: Optional[File] = Field(default=None, description="Preview audio generated with the cloned voice (if preview_text was provided)")


class App(BaseApp):
    """Inworld Voice Cloning app."""

    async def setup(self):
        self.logger = logging.getLogger(__name__)
        get_api_key()
        self.logger.info("Inworld Voice Cloning app initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        self.logger.info(f"Cloning voice: {input_data.display_name}")

        with open(input_data.audio.path, "rb") as f:
            audio_data = f.read()

        self.logger.info(f"Audio sample size: {len(audio_data)} bytes")

        result = await clone_voice(
            display_name=input_data.display_name,
            audio_data=audio_data,
            lang_code=input_data.language,
            description=input_data.description,
            remove_background_noise=input_data.remove_background_noise,
            logger=self.logger,
        )

        voice = result.get("voice", {})
        voice_id = voice.get("voiceId", "")

        self.logger.info(f"Voice cloned successfully: {voice_id}")

        preview_file = None
        if input_data.preview_text:
            self.logger.info(f"Generating preview with cloned voice: {voice_id}")
            preview_path = await text_to_speech(
                text=input_data.preview_text,
                voice_id=voice_id,
                model_id="inworld-tts-2",
                logger=self.logger,
            )
            preview_file = File(path=preview_path)

        return AppOutput(
            voice_id=voice_id,
            display_name=voice.get("displayName", input_data.display_name),
            language=voice.get("langCode", input_data.language),
            preview=preview_file,
        )

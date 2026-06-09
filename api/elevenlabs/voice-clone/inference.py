"""
ElevenLabs Voice Clone

Instantly clone a voice from audio samples.
The cloned voice ID can be used with ElevenLabs TTS, voice changer,
and text-to-dialogue apps.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, AudioMeta
from pydantic import Field
from typing import Optional, List
import logging

from .elevenlabs_helper import (
    clone_voice, get_api_key, get_voice_id,
    text_to_speech, get_audio_duration,
)


class AppInput(BaseAppInput):
    """Input schema for ElevenLabs Voice Clone."""

    audio: File = Field(
        description="Audio sample of the voice to clone (WAV or MP3, 1-2 minutes of clear speech recommended).",
    )
    name: str = Field(
        description="Name for the cloned voice (e.g. 'My Custom Voice').",
    )
    description: Optional[str] = Field(
        default=None,
        description="Optional description of the voice characteristics.",
    )
    remove_background_noise: bool = Field(
        default=True,
        description="Remove background noise from the audio sample before cloning.",
    )
    preview_text: Optional[str] = Field(
        default=None,
        description="Optional text to generate a preview with the cloned voice. If provided, returns a preview audio file.",
        max_length=500,
    )


class AppOutput(BaseAppOutput):
    """Output schema for ElevenLabs Voice Clone."""

    voice_id: str = Field(description="The cloned voice ID — use this with ElevenLabs TTS, voice changer, or text-to-dialogue apps")
    name: str = Field(description="Display name of the cloned voice")
    preview: Optional[File] = Field(default=None, description="Preview audio generated with the cloned voice (if preview_text was provided)")


class App(BaseApp):
    """ElevenLabs Voice Clone app."""

    async def setup(self):
        self.logger = logging.getLogger(__name__)
        get_api_key()
        self.logger.info("ElevenLabs Voice Clone app initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        self.logger.info(f"Cloning voice: {input_data.name}")

        result = clone_voice(
            name=input_data.name,
            files=[input_data.audio.path],
            description=input_data.description,
            remove_background_noise=input_data.remove_background_noise,
            logger=self.logger,
        )

        voice_id = result["voice_id"]
        self.logger.info(f"Voice cloned successfully: {voice_id}")

        preview_file = None
        if input_data.preview_text:
            self.logger.info(f"Generating preview with cloned voice")
            preview_path = text_to_speech(
                text=input_data.preview_text,
                voice_id=voice_id,
                model_id="eleven_v3",
                logger=self.logger,
            )
            duration = get_audio_duration(preview_path, self.logger)
            preview_file = File(path=preview_path)

        return AppOutput(
            voice_id=voice_id,
            name=result["name"],
            preview=preview_file,
        )

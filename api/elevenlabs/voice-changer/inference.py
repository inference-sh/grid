"""
ElevenLabs Voice Changer (Speech to Speech)

Transform the voice in an audio file to sound like a different voice
while preserving the original speech content.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, AudioMeta
from pydantic import Field
from typing import Literal
import logging

from .elevenlabs_helper import speech_to_speech, get_api_key, get_audio_duration, get_voice_id


class AppInput(BaseAppInput):
    """Input schema for ElevenLabs Voice Changer."""

    audio: File = Field(
        description="Input audio file containing speech to transform.",
    )
    voice: Literal[
        "adam", "alice", "aria", "bella", "bill", "brian", "callum", "charlie",
        "chris", "daniel", "eric", "george", "harry", "jessica", "laura", "liam",
        "lily", "matilda", "river", "roger", "sarah", "will",
    ] = Field(
        default="george",
        description="Target voice for transformation.",
    )
    model: Literal[
        "eleven_multilingual_sts_v2",
        "eleven_english_sts_v2",
    ] = Field(
        default="eleven_multilingual_sts_v2",
        description="Model to use. multilingual supports 70+ languages.",
    )
    output_format: Literal[
        "mp3_44100_128",
        "mp3_44100_192",
        "pcm_16000",
        "pcm_22050",
        "pcm_24000",
        "pcm_44100",
    ] = Field(
        default="mp3_44100_128",
        description="Audio output format.",
    )


class AppOutput(BaseAppOutput):
    """Output schema for ElevenLabs Voice Changer."""
    audio: File = Field(description="Transformed audio file with new voice")


class App(BaseApp):
    """ElevenLabs Voice Changer app implementation."""

    async def setup(self):
        """Initialize the application."""
        self.logger = logging.getLogger(__name__)
        get_api_key()
        self.logger.info("ElevenLabs Voice Changer app initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        """Transform voice in audio."""
        self.logger.info(f"Transforming voice to: {input_data.voice}")

        # Get input duration for billing (priced per minute)
        input_duration = get_audio_duration(input_data.audio.path, self.logger)

        audio_path = speech_to_speech(
            audio_path=input_data.audio.path,
            voice_id=get_voice_id(input_data.voice),
            model_id=input_data.model,
            output_format=input_data.output_format,
            logger=self.logger,
        )

        return AppOutput(
            audio=File(path=audio_path),
            output_meta=OutputMeta(
                inputs=[AudioMeta(
                    seconds=input_duration,
                    extra={"model": input_data.model}
                )],
                outputs=[AudioMeta(seconds=input_duration)]
            )
        )

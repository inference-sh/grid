"""
ElevenLabs Text to Speech

High-quality text-to-speech using ElevenLabs multilingual models.
Supports 32 languages with natural-sounding voices.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, TextMeta, AudioMeta
from pydantic import Field
from typing import Literal
import logging

from .elevenlabs_helper import text_to_speech, get_api_key, get_voice_id


class AppInput(BaseAppInput):
    """Input schema for ElevenLabs TTS."""

    text: str = Field(
        description="Text to convert to speech (max 40,000 characters).",
        max_length=40000,
    )
    voice: Literal[
        "adam",      # American male, dominant/firm
        "alice",     # British female, clear/engaging
        "aria",      # American female, expressive
        "bella",     # American female, professional/warm
        "bill",      # American male, wise/mature
        "brian",     # American male, deep/comforting
        "callum",    # American male, husky
        "charlie",   # Australian male, deep/energetic
        "chris",     # American male, charming
        "daniel",    # British male, broadcaster
        "eric",      # American male, smooth/trustworthy
        "george",    # British male, warm storyteller
        "harry",     # American male, fierce/rough
        "jessica",   # American female, playful/bright
        "laura",     # American female, quirky/sassy
        "liam",      # American male, energetic
        "lily",      # British female, velvety
        "matilda",   # American female, professional
        "river",     # American neutral, calm/informative
        "roger",     # American male, laid-back
        "sarah",     # American female, confident
        "will",      # American male, relaxed
    ] = Field(
        default="george",
        description="Voice to use for speech generation.",
    )
    model: Literal[
        "eleven_multilingual_v2",
        "eleven_turbo_v2_5",
        "eleven_flash_v2_5",
    ] = Field(
        default="eleven_multilingual_v2",
        description="Model to use. multilingual_v2 is highest quality, turbo/flash are faster with lower latency.",
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
        description="Audio output format. mp3_44100_128 is standard quality MP3.",
    )
    stability: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Voice stability (0-1). Higher = more consistent, lower = more expressive.",
    )
    similarity_boost: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Similarity boost (0-1). Higher = closer to original voice.",
    )
    style: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Style exaggeration (0-1). Increases expressiveness but may reduce stability.",
    )
    use_speaker_boost: bool = Field(
        default=True,
        description="Enable speaker boost for enhanced clarity.",
    )


class AppOutput(BaseAppOutput):
    """Output schema for ElevenLabs TTS."""
    audio: File = Field(description="Generated speech audio file")


class App(BaseApp):
    """ElevenLabs TTS app implementation."""

    async def setup(self):
        """Initialize the application."""
        self.logger = logging.getLogger(__name__)
        get_api_key()
        self.logger.info("ElevenLabs TTS app initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        """Generate speech from text."""
        self.logger.info(f"Generating speech: {len(input_data.text)} characters")
        self.logger.info(f"Voice: {input_data.voice}, Model: {input_data.model}")

        voice_settings = {
            "stability": input_data.stability,
            "similarity_boost": input_data.similarity_boost,
            "style": input_data.style,
            "use_speaker_boost": input_data.use_speaker_boost,
        }

        audio_path = text_to_speech(
            text=input_data.text,
            voice_id=get_voice_id(input_data.voice),
            model_id=input_data.model,
            output_format=input_data.output_format,
            voice_settings=voice_settings,
            logger=self.logger,
        )

        return AppOutput(
            audio=File(path=audio_path),
            output_meta=OutputMeta(
                inputs=[TextMeta(text=input_data.text)],
                outputs=[AudioMeta(seconds=0.0)]
            )
        )

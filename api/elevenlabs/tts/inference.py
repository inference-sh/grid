"""
ElevenLabs Text to Speech

High-quality text-to-speech using ElevenLabs models.
v3 supports 70+ languages with audio tags for emotion/style control.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, AudioMeta
from pydantic import Field
from typing import Literal, Optional
import logging

from .elevenlabs_helper import text_to_speech, get_api_key, get_voice_id, get_audio_duration


class AppInput(BaseAppInput):
    """Input schema for ElevenLabs TTS."""

    text: str = Field(
        description="Text to convert to speech. Max 40,000 characters for v2 models, 5,000 for v3.",
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
        description="Premade voice to use. Ignored if voice_id is provided.",
    )
    voice_id: Optional[str] = Field(
        default=None,
        description="Custom voice ID (e.g. from elevenlabs/voice-clone). Overrides the voice field when provided.",
    )
    model: Literal[
        "eleven_v3",
        "eleven_multilingual_v2",
        "eleven_turbo_v2_5",
        "eleven_flash_v2_5",
    ] = Field(
        default="eleven_v3",
        description="Model to use. v3 is most expressive with 70+ languages and audio tag support, multilingual_v2 is high quality, turbo/flash are faster with lower latency.",
    )
    audio_tags: bool = Field(
        default=False,
        description="Enable audio tags in text for emotion/style control (v3 only). Use tags like [laughs], [whispers], [excited], [sad], [slow], [fast], [shouts], [sighs] inline in your text.",
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
        max_chars = 5000 if input_data.model == "eleven_v3" else 40000
        if len(input_data.text) > max_chars:
            raise ValueError(f"Text exceeds {max_chars} character limit for {input_data.model}")

        if input_data.audio_tags and input_data.model != "eleven_v3":
            raise ValueError("Audio tags are only supported with the eleven_v3 model")

        resolved_voice_id = input_data.voice_id if input_data.voice_id else get_voice_id(input_data.voice)

        self.logger.info(f"Generating speech: {len(input_data.text)} characters")
        self.logger.info(f"Voice ID: {resolved_voice_id}, Model: {input_data.model}")

        voice_settings = {
            "stability": input_data.stability,
            "similarity_boost": input_data.similarity_boost,
            "style": input_data.style,
            "use_speaker_boost": input_data.use_speaker_boost,
        }

        audio_path = text_to_speech(
            text=input_data.text,
            voice_id=resolved_voice_id,
            model_id=input_data.model,
            output_format=input_data.output_format,
            voice_settings=voice_settings,
            logger=self.logger,
        )

        duration = get_audio_duration(audio_path, self.logger)

        return AppOutput(
            audio=File(path=audio_path),
            output_meta=OutputMeta(
                inputs=[],
                outputs=[AudioMeta(
                    seconds=duration,
                    extra={"characters": len(input_data.text), "model": input_data.model}
                )]
            )
        )

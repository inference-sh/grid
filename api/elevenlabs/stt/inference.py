"""
ElevenLabs Speech to Text (Scribe)

High-accuracy speech transcription with speaker diarization
and audio event detection. Supports 90+ languages.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, AudioMeta
from pydantic import Field
from typing import Optional, Literal, List
import logging

from .elevenlabs_helper import speech_to_text, get_api_key, get_audio_duration


class AppInput(BaseAppInput):
    """Input schema for ElevenLabs STT."""

    audio: File = Field(
        description="Audio file to transcribe (MP3, WAV, FLAC, OGG, AAC, M4A).",
    )
    model: Literal["scribe_v1", "scribe_v2"] = Field(
        default="scribe_v2",
        description="Model version. scribe_v2 is latest with improved accuracy.",
    )
    language_code: Optional[str] = Field(
        default=None,
        description="Language code (e.g., 'eng', 'spa', 'fra'). Leave empty for auto-detection.",
    )
    diarize: bool = Field(
        default=False,
        description="Enable speaker diarization to identify who is speaking.",
    )
    tag_audio_events: bool = Field(
        default=False,
        description="Tag audio events like laughter, applause, music, etc.",
    )


class AppOutput(BaseAppOutput):
    """Output schema for ElevenLabs STT."""
    text: str = Field(description="Full transcription text")
    language_code: Optional[str] = Field(default=None, description="Detected language code")
    language_probability: Optional[float] = Field(default=None, description="Language detection confidence")
    words: Optional[List[dict]] = Field(default=None, description="Word-level timestamps and speaker info")


class App(BaseApp):
    """ElevenLabs STT app implementation."""

    async def setup(self):
        """Initialize the application."""
        self.logger = logging.getLogger(__name__)
        get_api_key()
        self.logger.info("ElevenLabs STT app initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        """Transcribe audio to text."""
        self.logger.info(f"Transcribing audio: {input_data.audio.path}")
        self.logger.info(f"Model: {input_data.model}, Diarize: {input_data.diarize}")

        result = speech_to_text(
            audio=input_data.audio.path,
            model_id=input_data.model,
            language_code=input_data.language_code,
            diarize=input_data.diarize,
            tag_audio_events=input_data.tag_audio_events,
            logger=self.logger,
        )

        # Get duration from API response word timestamps, fallback to ffprobe
        duration_seconds = 0.0
        words = result.get("words", [])
        if words and len(words) > 0:
            # Use end time of last word as duration
            duration_seconds = float(words[-1].get("end", 0))
            self.logger.info(f"Duration from API: {duration_seconds:.2f}s")
        if duration_seconds == 0.0:
            duration_seconds = get_audio_duration(input_data.audio.path, self.logger)

        return AppOutput(
            text=result.get("text", ""),
            language_code=result.get("language_code"),
            language_probability=result.get("language_probability"),
            words=words,
            output_meta=OutputMeta(
                inputs=[AudioMeta(
                    seconds=duration_seconds,
                    extra={"model": input_data.model}
                )],
                outputs=[]
            )
        )

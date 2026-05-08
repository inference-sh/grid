"""
Inworld Speech to Text

Multi-provider speech transcription with optional word timestamps
and voice profile identification.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, AudioMeta
from pydantic import Field
from typing import Optional, List
import logging

from .inworld_helper import speech_to_text, get_api_key, get_audio_duration


class AppInput(BaseAppInput):
    """Input schema for Inworld STT."""

    audio: File = Field(
        description="Audio file to transcribe (MP3, WAV, FLAC, OGG, PCM).",
    )
    language: Optional[str] = Field(
        default=None,
        description="BCP-47 language code (e.g. 'en-US', 'ja-JP'). Auto-detected if omitted.",
    )
    include_word_timestamps: bool = Field(
        default=False,
        description="Include per-word timing information in the output.",
    )


class AppOutput(BaseAppOutput):
    """Output schema for Inworld STT."""
    text: str = Field(description="Full transcription text")
    words: Optional[List[dict]] = Field(default=None, description="Word-level timestamps with confidence scores")


class App(BaseApp):
    """Inworld STT app implementation."""

    async def setup(self):
        self.logger = logging.getLogger(__name__)
        get_api_key()
        self.logger.info("Inworld STT app initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        self.logger.info(f"Transcribing audio: {input_data.audio.path}")

        result = await speech_to_text(
            audio_path=input_data.audio.path,
            model_id="inworld/inworld-stt-1",
            language=input_data.language,
            include_word_timestamps=input_data.include_word_timestamps,
            logger=self.logger,
        )

        transcription = result.get("transcription", {})
        transcript = transcription.get("transcript", "")
        words = transcription.get("wordTimestamps")

        # Get duration from usage or word timestamps, fallback to ffprobe
        usage = result.get("usage", {})
        duration_ms = usage.get("transcribedAudioMs", 0)
        duration_seconds = duration_ms / 1000.0 if duration_ms else 0.0

        if duration_seconds == 0.0 and words:
            last_word = words[-1]
            duration_seconds = last_word.get("endTimeMs", 0) / 1000.0

        if duration_seconds == 0.0:
            duration_seconds = get_audio_duration(input_data.audio.path, self.logger)

        self.logger.info(f"Transcription: {len(transcript)} chars, {duration_seconds:.2f}s audio")

        return AppOutput(
            text=transcript,
            words=words,
            output_meta=OutputMeta(
                inputs=[AudioMeta(
                    seconds=duration_seconds,
                    extra={"model": "inworld/inworld-stt-1"},
                )],
                outputs=[],
            ),
        )

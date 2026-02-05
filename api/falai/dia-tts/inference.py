"""
Dia TTS - Text to Speech with Voice Cloning

Generates realistic dialogue from transcripts with emotion control
and natural nonverbals like laughter and throat clearing.

Supports two modes:
- Basic TTS: Just provide text
- Voice Clone: Provide text + reference audio + reference text
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, AudioMeta, TextMeta
from pydantic import Field
from typing import Optional
import logging

from .fal_helper import setup_fal_client, run_fal_model, download_file

# Suppress noisy httpx polling logs
logging.getLogger("httpx").setLevel(logging.WARNING)


class AppInput(BaseAppInput):
    """Input schema for Dia TTS."""
    text: str = Field(
        description="The text to convert to speech. Use [S1], [S2] for multi-speaker dialogue and (laughs), (sighs) for nonverbals."
    )
    ref_audio: Optional[File] = Field(
        default=None,
        description="Reference audio file for voice cloning. If provided, ref_text is also required."
    )
    ref_text: Optional[str] = Field(
        default=None,
        description="Transcript of the reference audio. Required when using voice cloning."
    )


class AppOutput(BaseAppOutput):
    """Output schema for Dia TTS."""
    audio: File = Field(description="The generated speech audio")


class App(BaseApp):
    """Dia TTS app implementation."""

    async def setup(self):
        """Initialize the application."""
        self.logger = logging.getLogger(__name__)
        self.tts_model_id = "fal-ai/dia-tts"
        self.clone_model_id = "fal-ai/dia-tts/voice-clone"
        self.logger.info("Dia TTS app initialized")

    def _get_model_id(self, input_data: AppInput) -> str:
        """Select endpoint based on input."""
        if input_data.ref_audio:
            return self.clone_model_id
        return self.tts_model_id

    def _build_request(self, input_data: AppInput) -> dict:
        """Build request payload."""
        request = {"text": input_data.text}

        if input_data.ref_audio:
            if not input_data.ref_text:
                raise ValueError("ref_text is required when using voice cloning (ref_audio provided)")
            request["ref_audio_url"] = input_data.ref_audio.uri
            request["ref_text"] = input_data.ref_text

        return request

    async def run(self, input_data: AppInput) -> AppOutput:
        """Run text-to-speech inference."""
        try:
            setup_fal_client()

            model_id = self._get_model_id(input_data)
            mode = "voice-clone" if input_data.ref_audio else "basic-tts"
            self.logger.info(f"Mode: {mode}, Model: {model_id}")
            self.logger.info(f"Processing text: {input_data.text[:100]}...")

            request_data = self._build_request(input_data)
            result = run_fal_model(model_id, request_data, self.logger)

            # Download the generated audio
            audio_url = result["audio"]["url"]
            audio_path = download_file(audio_url, suffix=".wav", logger=self.logger)

            return AppOutput(
                audio=File(path=audio_path),
                output_meta=OutputMeta(
                    inputs=[TextMeta(text=input_data.text)],
                    outputs=[AudioMeta(seconds=0.0)]
                )
            )

        except Exception as e:
            self.logger.error(f"Error: {e}")
            raise RuntimeError(f"TTS generation failed: {str(e)}")

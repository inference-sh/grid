"""
Kling Create Voice

Clone a voice from an audio or video sample using Kling's voice creation API.
Returns a voice ID that can be used with the kling-lipsync text-to-video mode.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta
from pydantic import Field
import logging

from .fal_helper import setup_fal_client, run_fal_model

logging.getLogger("httpx").setLevel(logging.WARNING)


ENDPOINT = "fal-ai/kling-video/create-voice"


class AppInput(BaseAppInput):
    """Input schema for Kling Create Voice."""

    voice: File = Field(
        description="Audio (.mp3/.wav) or video (.mp4/.mov) sample for voice cloning. Must be 10-300 seconds, max 100MB.",
    )


class AppOutput(BaseAppOutput):
    """Output schema for Kling Create Voice."""

    voice_id: str = Field(description="The created voice identifier, usable with kling-lipsync text mode.")


class App(BaseApp):
    """Kling Create Voice application."""

    async def setup(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Kling Create Voice initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        try:
            setup_fal_client()

            self.logger.info("Creating voice clone from audio/video sample")

            request_data = {
                "voice_url": input_data.voice.uri,
            }

            result = run_fal_model(ENDPOINT, request_data, self.logger)

            voice_id = result["voice_id"]
            self.logger.info(f"Voice created successfully: {voice_id}")

            return AppOutput(
                voice_id=voice_id,
                output_meta=OutputMeta(),
            )

        except Exception as e:
            self.logger.error(f"Error during voice creation: {e}")
            raise RuntimeError(f"Voice creation failed: {str(e)}")

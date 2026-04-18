"""
Kling v2.0 Master Image-to-Video

Generate videos from a starting image using Kling v2.0 Master model via fal.ai.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta
from pydantic import Field
from typing import Optional
from enum import Enum
import logging

from .fal_helper import setup_fal_client, run_fal_model, download_video

logging.getLogger("httpx").setLevel(logging.WARNING)

MODEL_ENDPOINT = "fal-ai/kling-video/v2/master/image-to-video"


class DurationEnum(str, Enum):
    s5 = "5"
    s10 = "10"


class AppInput(BaseAppInput):
    """Input schema for Kling v2.0 Master Image-to-Video."""

    prompt: str = Field(
        description="The text prompt describing the desired motion and action for the video.",
        examples=["The woman slowly turns her head and smiles at the camera"],
    )
    image: File = Field(
        description="Starting frame image to animate. Supported formats: JPEG, PNG, WebP.",
    )
    duration: DurationEnum = Field(
        default=DurationEnum.s5,
        description="Duration of the video in seconds (5 or 10).",
    )
    cfg_scale: Optional[float] = Field(
        default=None,
        description="Classifier-free guidance scale. Controls how closely the video follows the prompt.",
    )
    negative_prompt: Optional[str] = Field(
        default=None,
        description="Negative prompt to guide generation away from unwanted content.",
    )


class AppOutput(BaseAppOutput):
    """Output schema for Kling v2.0 Master Image-to-Video."""

    video: File = Field(description="The generated video file.")


class App(BaseApp):
    """Kling v2.0 Master Image-to-Video application."""

    async def setup(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Kling v2.0 Master I2V initialized")

    def _build_request(self, input_data: AppInput) -> dict:
        request = {
            "prompt": input_data.prompt,
            "image_url": input_data.image.uri,
            "duration": input_data.duration.value,
        }
        if input_data.cfg_scale is not None:
            request["cfg_scale"] = input_data.cfg_scale
        if input_data.negative_prompt:
            request["negative_prompt"] = input_data.negative_prompt
        return request

    async def run(self, input_data: AppInput) -> AppOutput:
        try:
            if not input_data.image.exists():
                raise RuntimeError(f"Input image does not exist at path: {input_data.image.path}")

            setup_fal_client()

            self.logger.info(f"Generating video with prompt: {input_data.prompt[:100]}...")
            self.logger.info(f"Settings: {input_data.duration.value}s")

            request_data = self._build_request(input_data)
            result = run_fal_model(MODEL_ENDPOINT, request_data, self.logger)

            video_url = result["video"]["url"]
            video_path = download_video(video_url, self.logger)

            # I2V dimensions depend on input image; default to 16:9
            width, height = 1280, 720
            seconds = float(input_data.duration.value)

            output_meta = OutputMeta(
                outputs=[
                    VideoMeta(
                        width=width,
                        height=height,
                        seconds=seconds,
                        fps=24,
                    )
                ]
            )

            return AppOutput(
                video=File(path=video_path),
                output_meta=output_meta,
            )

        except Exception as e:
            self.logger.error(f"Error during video generation: {e}")
            raise RuntimeError(f"Video generation failed: {str(e)}")

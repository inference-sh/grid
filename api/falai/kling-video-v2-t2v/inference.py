"""
Kling v2.0 Master Text-to-Video

Generate videos from text prompts using Kling v2.0 Master model via fal.ai.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta
from pydantic import Field
from typing import Optional
from enum import Enum
import logging

from .fal_helper import setup_fal_client, run_fal_model, download_video

logging.getLogger("httpx").setLevel(logging.WARNING)

MODEL_ENDPOINT = "fal-ai/kling-video/v2/master/text-to-video"

DEFAULT_DURATION_SECONDS = 5.0

VIDEO_DIMENSIONS = {
    "16:9": (1280, 720),
    "9:16": (720, 1280),
    "1:1": (720, 720),
}


class AspectRatioEnum(str, Enum):
    ratio_16_9 = "16:9"
    ratio_9_16 = "9:16"
    ratio_1_1 = "1:1"


class DurationEnum(str, Enum):
    s5 = "5"
    s10 = "10"


class AppInput(BaseAppInput):
    """Input schema for Kling v2.0 Master Text-to-Video."""

    prompt: str = Field(
        description="The text prompt describing the video to generate.",
        examples=["A golden retriever running through a field of sunflowers at sunset"],
    )
    aspect_ratio: AspectRatioEnum = Field(
        default=AspectRatioEnum.ratio_16_9,
        description="Aspect ratio of the generated video.",
    )
    duration: DurationEnum = Field(
        default=DurationEnum.s5,
        description="Duration of the video in seconds (5 or 10).",
    )
    cfg_scale: float = Field(
        default=0.5,
        description="Classifier-free guidance scale. Controls how closely the video follows the prompt.",
    )
    negative_prompt: Optional[str] = Field(
        default=None,
        description="Negative prompt to guide generation away from unwanted content.",
    )


class AppOutput(BaseAppOutput):
    """Output schema for Kling v2.0 Master Text-to-Video."""

    video: File = Field(description="The generated video file.")


class App(BaseApp):
    """Kling v2.0 Master Text-to-Video application."""

    async def setup(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Kling v2.0 Master T2V initialized")

    def _build_request(self, input_data: AppInput) -> dict:
        request = {
            "prompt": input_data.prompt,
            "aspect_ratio": input_data.aspect_ratio.value,
            "duration": input_data.duration.value,
            "cfg_scale": input_data.cfg_scale,
        }
        if input_data.negative_prompt:
            request["negative_prompt"] = input_data.negative_prompt
        return request

    async def run(self, input_data: AppInput) -> AppOutput:
        try:
            setup_fal_client()

            self.logger.info(f"Generating video with prompt: {input_data.prompt[:100]}...")
            self.logger.info(
                f"Settings: {input_data.aspect_ratio.value}, {input_data.duration.value}s, cfg={input_data.cfg_scale}"
            )

            request_data = self._build_request(input_data)
            result = run_fal_model(MODEL_ENDPOINT, request_data, self.logger)

            video_url = result["video"]["url"]
            video_path = download_video(video_url, self.logger)

            width, height = VIDEO_DIMENSIONS.get(input_data.aspect_ratio.value, (1280, 720))
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

"""
Kling Video v2.1 Master Text-to-Video

Generate videos from text prompts using Kling Video v2.1 Master model via fal.ai.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta
from pydantic import Field
from typing import Optional
from enum import Enum
import logging

from .fal_helper import setup_fal_client, run_fal_model, download_video

logging.getLogger("httpx").setLevel(logging.WARNING)


class AspectRatioEnum(str, Enum):
    """Video aspect ratio options."""
    ratio_16_9 = "16:9"
    ratio_9_16 = "9:16"
    ratio_1_1 = "1:1"


class DurationEnum(str, Enum):
    """Video duration options in seconds."""
    s5 = "5"
    s10 = "10"


VIDEO_DIMENSIONS = {
    "16:9": (1280, 720),
    "9:16": (720, 1280),
    "1:1": (720, 720),
}

MODEL_ENDPOINT = "fal-ai/kling-video/v2.1/master/text-to-video"


class AppInput(BaseAppInput):
    """Input schema for Kling Video v2.1 Master Text-to-Video."""

    prompt: str = Field(
        description="The text prompt describing the video to generate.",
        examples=[
            "A golden retriever running through a sunlit meadow with butterflies."
        ],
    )
    cfg_scale: float = Field(
        default=0.5,
        description="Classifier-free guidance scale. Higher values follow the prompt more strictly.",
    )
    aspect_ratio: AspectRatioEnum = Field(
        default=AspectRatioEnum.ratio_16_9,
        description="The aspect ratio of the generated video.",
    )
    negative_prompt: str = Field(
        default="blur, distort, and low quality",
        description="Negative prompt to guide what to avoid in generation.",
    )
    duration: DurationEnum = Field(
        default=DurationEnum.s5,
        description="Duration of the video in seconds.",
    )


class AppOutput(BaseAppOutput):
    """Output schema for Kling Video v2.1 Master Text-to-Video."""

    video: File = Field(description="The generated video file.")
    seed: Optional[int] = Field(default=None, description="The seed used for generation.")


class App(BaseApp):
    """Kling Video v2.1 Master Text-to-Video application."""

    async def setup(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Kling Video v2.1 Master T2V initialized")

    def _build_request(self, input_data: AppInput) -> dict:
        request = {
            "prompt": input_data.prompt,
            "cfg_scale": input_data.cfg_scale,
            "aspect_ratio": input_data.aspect_ratio.value,
            "negative_prompt": input_data.negative_prompt,
            "duration": input_data.duration.value,
        }
        return request

    async def run(self, input_data: AppInput) -> AppOutput:
        try:
            setup_fal_client()

            self.logger.info(
                f"Generating video with prompt: {input_data.prompt[:100]}..."
            )
            self.logger.info(
                f"Settings: {input_data.aspect_ratio.value}, {input_data.duration.value}s, cfg={input_data.cfg_scale}"
            )

            request_data = self._build_request(input_data)
            result = run_fal_model(MODEL_ENDPOINT, request_data, self.logger)

            video_url = result["video"]["url"]
            video_path = download_video(video_url, self.logger)

            width, height = VIDEO_DIMENSIONS.get(
                input_data.aspect_ratio.value, (1280, 720)
            )
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
                seed=result.get("seed"),
                output_meta=output_meta,
            )

        except Exception as e:
            self.logger.error(f"Error during video generation: {e}")
            raise RuntimeError(f"Video generation failed: {str(e)}")

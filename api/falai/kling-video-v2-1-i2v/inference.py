"""
Kling Video v2.1 Image-to-Video

Generate videos from images using Kling Video v2.1 model via fal.ai.
Supports master, pro, and standard tiers.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta
from pydantic import Field
from typing import Optional
from enum import Enum
import logging

from .fal_helper import setup_fal_client, run_fal_model, download_video

logging.getLogger("httpx").setLevel(logging.WARNING)


class TierEnum(str, Enum):
    """Quality tier — master is highest quality, pro is balanced, standard is fastest."""
    master = "master"
    pro = "pro"
    standard = "standard"


class DurationEnum(str, Enum):
    """Video duration options in seconds."""
    s5 = "5"
    s10 = "10"


ENDPOINT_BY_TIER = {
    TierEnum.master: "fal-ai/kling-video/v2.1/master/image-to-video",
    TierEnum.pro: "fal-ai/kling-video/v2.1/pro/image-to-video",
    TierEnum.standard: "fal-ai/kling-video/v2.1/standard/image-to-video",
}

# i2v does not take aspect_ratio — dimensions come from the input image.
# Default to 16:9 for output metadata.
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720


class AppInput(BaseAppInput):
    """Input schema for Kling Video v2.1 Image-to-Video."""

    prompt: str = Field(
        description="The text prompt describing the desired motion and scene.",
        examples=[
            "The camera slowly zooms in as the subject smiles and waves."
        ],
    )
    image: File = Field(
        description="The source image to animate.",
    )
    cfg_scale: float = Field(
        default=0.5,
        description="Classifier-free guidance scale. Higher values follow the prompt more strictly.",
    )
    negative_prompt: str = Field(
        default="blur, distort, and low quality",
        description="Negative prompt to guide what to avoid in generation.",
    )
    duration: DurationEnum = Field(
        default=DurationEnum.s5,
        description="Duration of the video in seconds.",
    )
    tier: TierEnum = Field(
        default=TierEnum.master,
        description="Quality tier. Master is highest quality, pro is balanced, standard is fastest.",
    )


class AppOutput(BaseAppOutput):
    """Output schema for Kling Video v2.1 Image-to-Video."""

    video: File = Field(description="The generated video file.")
    seed: Optional[int] = Field(default=None, description="The seed used for generation.")


class App(BaseApp):
    """Kling Video v2.1 Image-to-Video application."""

    async def setup(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Kling Video v2.1 I2V initialized")

    def _build_request(self, input_data: AppInput) -> dict:
        request = {
            "prompt": input_data.prompt,
            "image_url": input_data.image.uri,
            "cfg_scale": input_data.cfg_scale,
            "negative_prompt": input_data.negative_prompt,
            "duration": input_data.duration.value,
        }
        return request

    async def run(self, input_data: AppInput) -> AppOutput:
        try:
            setup_fal_client()

            model_id = ENDPOINT_BY_TIER[input_data.tier]
            self.logger.info(
                f"Generating video ({input_data.tier.value}) with prompt: {input_data.prompt[:100]}..."
            )
            self.logger.info(
                f"Settings: {input_data.duration.value}s, cfg={input_data.cfg_scale}, endpoint={model_id}"
            )

            request_data = self._build_request(input_data)
            result = run_fal_model(model_id, request_data, self.logger)

            video_url = result["video"]["url"]
            video_path = download_video(video_url, self.logger)

            seconds = float(input_data.duration.value)

            output_meta = OutputMeta(
                outputs=[
                    VideoMeta(
                        width=DEFAULT_WIDTH,
                        height=DEFAULT_HEIGHT,
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

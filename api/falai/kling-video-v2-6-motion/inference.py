"""
Kling Video v2.6 Motion Control

Generate videos with motion control from a reference image and video
using Kling Video v2.6 model via fal.ai. Supports pro and standard tiers.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta
from pydantic import Field
from typing import Optional
from enum import Enum
import logging

from .fal_helper import setup_fal_client, run_fal_model, download_video

logging.getLogger("httpx").setLevel(logging.WARNING)


class TierEnum(str, Enum):
    """Model tier — pro has higher quality, standard is cheaper."""
    pro = "pro"
    standard = "standard"


class CharacterOrientationEnum(str, Enum):
    """Whether output orientation matches the reference image or video."""
    image = "image"
    video = "video"


ENDPOINT_BY_TIER = {
    TierEnum.pro: "fal-ai/kling-video/v2.6/pro/motion-control",
    TierEnum.standard: "fal-ai/kling-video/v2.6/standard/motion-control",
}


class AppInput(BaseAppInput):
    """Input schema for Kling Video v2.6 Motion Control."""

    image: File = Field(
        description="Reference image for the video. Supported formats: JPEG, PNG, WebP.",
    )
    video: File = Field(
        description="Reference video providing the motion to follow.",
    )
    prompt: Optional[str] = Field(
        default=None,
        description="Optional text prompt to guide the generation.",
    )
    character_orientation: CharacterOrientationEnum = Field(
        default=CharacterOrientationEnum.image,
        description="Whether the output orientation matches the reference image or video.",
    )
    keep_original_sound: bool = Field(
        default=True,
        description="Whether to keep the original sound from the reference video.",
    )
    tier: TierEnum = Field(
        default=TierEnum.pro,
        description="Model tier. Pro has higher quality ($0.112/s), standard is cheaper ($0.07/s).",
    )


class AppOutput(BaseAppOutput):
    """Output schema for Kling Video v2.6 Motion Control."""

    video: File = Field(description="The generated video file.")


class App(BaseApp):
    """Kling Video v2.6 Motion Control application."""

    async def setup(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Kling Video v2.6 Motion Control initialized")

    def _build_request(self, input_data: AppInput) -> dict:
        request = {
            "image_url": input_data.image.uri,
            "video_url": input_data.video.uri,
            "character_orientation": input_data.character_orientation.value,
            "keep_original_sound": input_data.keep_original_sound,
        }
        if input_data.prompt:
            request["prompt"] = input_data.prompt
        return request

    async def run(self, input_data: AppInput) -> AppOutput:
        try:
            if not input_data.image.exists():
                raise RuntimeError(f"Input image does not exist at path: {input_data.image.path}")
            if not input_data.video.exists():
                raise RuntimeError(f"Input video does not exist at path: {input_data.video.path}")

            setup_fal_client()

            model_id = ENDPOINT_BY_TIER[input_data.tier]
            self.logger.info(
                f"Generating motion control video ({input_data.tier.value}) endpoint={model_id}"
            )
            if input_data.prompt:
                self.logger.info(f"Prompt: {input_data.prompt[:100]}...")

            request_data = self._build_request(input_data)
            result = run_fal_model(model_id, request_data, self.logger)

            video_url = result["video"]["url"]
            video_path = download_video(video_url, self.logger)

            # Default to 16:9 dimensions for output metadata
            width, height = 1280, 720
            # Estimate seconds from result if available, default to 5
            seconds = 5.0

            output_meta = OutputMeta(
                outputs=[
                    VideoMeta(
                        width=width,
                        height=height,
                        seconds=seconds,
                        fps=24,
                        extra={
                            "tier": input_data.tier.value,
                        },
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

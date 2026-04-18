"""
Kling O3 Reference-to-Video

Generate videos from reference images using Kling O3 model via fal.ai.
Supports pro and standard tiers. Reference images in the prompt as @Image1, @Image2, etc.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta
from pydantic import Field
from typing import List, Optional
from enum import Enum
import logging

from .fal_helper import setup_fal_client, run_fal_model, download_video

logging.getLogger("httpx").setLevel(logging.WARNING)

VIDEO_DIMENSIONS = {
    "16:9": (1280, 720),
    "9:16": (720, 1280),
    "1:1": (720, 720),
}


class TierEnum(str, Enum):
    pro = "pro"
    standard = "standard"


class AspectRatioEnum(str, Enum):
    ratio_16_9 = "16:9"
    ratio_9_16 = "9:16"
    ratio_1_1 = "1:1"


class DurationEnum(str, Enum):
    s3 = "3"
    s4 = "4"
    s5 = "5"
    s6 = "6"
    s7 = "7"
    s8 = "8"
    s9 = "9"
    s10 = "10"
    s11 = "11"
    s12 = "12"
    s13 = "13"
    s14 = "14"
    s15 = "15"


class AppInput(BaseAppInput):
    """Input schema for Kling O3 Reference-to-Video."""

    prompt: Optional[str] = Field(
        default=None,
        description="The text prompt describing the video. Reference style images as @Image1, @Image2, etc.",
        examples=["A woman in the style of @Image1 walks through a garden"],
    )
    start_image: Optional[File] = Field(
        default=None,
        description="Optional starting frame image for the video.",
    )
    end_image: Optional[File] = Field(
        default=None,
        description="Optional ending frame image. The video will transition toward this image.",
    )
    image_urls: Optional[List[File]] = Field(
        default=None,
        description="Style reference images. Referenced in the prompt as @Image1, @Image2, etc.",
    )
    aspect_ratio: AspectRatioEnum = Field(
        default=AspectRatioEnum.ratio_16_9,
        description="Aspect ratio of the generated video.",
    )
    duration: DurationEnum = Field(
        default=DurationEnum.s5,
        description="Duration of the video in seconds (3-15).",
    )
    generate_audio: bool = Field(
        default=False,
        description="Whether to generate audio for the video.",
    )
    tier: TierEnum = Field(
        default=TierEnum.pro,
        description="Model tier. Pro offers higher quality, standard is faster and cheaper.",
    )


class AppOutput(BaseAppOutput):
    """Output schema for Kling O3 Reference-to-Video."""

    video: File = Field(description="The generated video file.")


class App(BaseApp):
    """Kling O3 Reference-to-Video application."""

    async def setup(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Kling O3 R2V initialized")

    def _build_request(self, input_data: AppInput) -> dict:
        request = {
            "aspect_ratio": input_data.aspect_ratio.value,
            "duration": input_data.duration.value,
            "generate_audio": input_data.generate_audio,
        }
        if input_data.prompt:
            request["prompt"] = input_data.prompt
        if input_data.start_image:
            request["start_image_url"] = input_data.start_image.uri
        if input_data.end_image:
            request["end_image_url"] = input_data.end_image.uri
        if input_data.image_urls:
            request["image_urls"] = [f.uri for f in input_data.image_urls]
        return request

    async def run(self, input_data: AppInput) -> AppOutput:
        try:
            if input_data.start_image and not input_data.start_image.exists():
                raise RuntimeError(f"Start image does not exist at path: {input_data.start_image.path}")
            if input_data.end_image and not input_data.end_image.exists():
                raise RuntimeError(f"End image does not exist at path: {input_data.end_image.path}")
            if input_data.image_urls:
                for idx, f in enumerate(input_data.image_urls):
                    if f and not f.exists():
                        raise RuntimeError(f"Reference image {idx + 1} does not exist at path: {f.path}")

            setup_fal_client()

            model_id = f"fal-ai/kling-video/o3/{input_data.tier.value}/reference-to-video"
            prompt_preview = input_data.prompt[:100] if input_data.prompt else "(no prompt)"
            self.logger.info(f"Generating video ({input_data.tier.value}) with prompt: {prompt_preview}...")
            self.logger.info(
                f"Settings: {input_data.aspect_ratio.value}, {input_data.duration.value}s, "
                f"refs={len(input_data.image_urls or [])}, endpoint={model_id}"
            )

            request_data = self._build_request(input_data)
            result = run_fal_model(model_id, request_data, self.logger)

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

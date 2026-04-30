"""
HappyHorse 1.0 Reference-to-Video - Alibaba Cloud Video Generation

Generate videos that preserve subject characters from up to 9 reference images,
driven by a text prompt describing the desired scene. Uses the HappyHorse 1.0
R2V model via DashScope API.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta, VideoResolution
from pydantic import Field
from typing import Optional, List
from enum import Enum
import logging

from .wan_video_helper import generate_video, extract_video_url, extract_usage, download_video


class ResolutionEnum(str, Enum):
    hd = "720P"
    fhd = "1080P"


class AspectRatioEnum(str, Enum):
    landscape = "16:9"
    portrait = "9:16"
    square = "1:1"
    standard = "4:3"
    standard_portrait = "3:4"


class AppInput(BaseAppInput):
    """Input schema for HappyHorse 1.0 Reference-to-Video."""

    prompt: str = Field(
        description="Text prompt describing the video scene. Use '[Image 1]', '[Image 2]' etc. to reference images in order. You must specify the object from the reference image, e.g. 'the woman in a red qipao in [Image 1]'. Up to 5000 non-Chinese characters."
    )
    reference_images: List[File] = Field(
        description="Reference images for characters/objects/scenes. 1-9 images. Formats: JPEG, JPG, PNG, WEBP. Shortest side at least 400px (720P+ recommended). Up to 10MB each."
    )
    resolution: ResolutionEnum = Field(
        default=ResolutionEnum.fhd,
        description="Video resolution: 720P or 1080P (default)."
    )
    ratio: AspectRatioEnum = Field(
        default=AspectRatioEnum.landscape,
        description="Aspect ratio of the generated video."
    )
    duration: int = Field(
        default=5,
        ge=3,
        le=15,
        description="Video duration in seconds (3-15)."
    )
    watermark: bool = Field(
        default=False,
        description="Add 'HappyHorse' watermark to bottom-right corner."
    )
    seed: Optional[int] = Field(
        default=None,
        ge=0,
        le=2147483647,
        description="Random seed for reproducibility (0-2147483647)."
    )


class AppOutput(BaseAppOutput):
    """Output schema for HappyHorse 1.0 Reference-to-Video."""

    video: File = Field(description="Generated video in MP4 format.")


class App(BaseApp):
    """HappyHorse 1.0 Reference-to-Video generation."""

    async def setup(self, metadata):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.model = "happyhorse-1.0-r2v"
        self.logger.info("HappyHorse 1.0 R2V initialized")

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        if not input_data.reference_images or len(input_data.reference_images) < 1:
            raise RuntimeError("At least 1 reference image is required.")
        if len(input_data.reference_images) > 9:
            raise RuntimeError("Maximum 9 reference images allowed.")

        media = []
        for img in input_data.reference_images:
            media.append({"type": "reference_image", "url": img.uri})

        num_images = len(input_data.reference_images)
        self.logger.info(f"R2V: {num_images} reference images")
        self.logger.info(f"Resolution: {input_data.resolution.value}, Ratio: {input_data.ratio.value}, Duration: {input_data.duration}s")

        input_payload = {
            "prompt": input_data.prompt,
            "media": media,
        }

        parameters = {
            "resolution": input_data.resolution.value,
            "ratio": input_data.ratio.value,
            "duration": input_data.duration,
            "watermark": input_data.watermark,
        }
        if input_data.seed is not None:
            parameters["seed"] = input_data.seed

        result = await generate_video(
            model=self.model,
            input_data=input_payload,
            parameters=parameters,
            logger=self.logger,
        )

        video_url = extract_video_url(result)
        video_path = await download_video(video_url, logger=self.logger)

        usage = extract_usage(result)
        sr = usage.get("SR", 1080)
        duration = usage.get("duration", input_data.duration)
        ratio = usage.get("ratio", input_data.ratio.value)

        resolution_map = {
            720: VideoResolution.VIDEO_RES720_P,
            1080: VideoResolution.VIDEO_RES1080_P,
        }

        self.logger.info(f"Video generated: {sr}P, {duration}s, ratio={ratio}")

        return AppOutput(
            video=File(path=video_path),
            output_meta=OutputMeta(
                outputs=[
                    VideoMeta(
                        resolution=resolution_map.get(sr, VideoResolution.VIDEO_RES1080_P),
                        seconds=float(duration),
                        extra={"model": self.model, "ratio": ratio, "reference_count": num_images},
                    )
                ]
            ),
        )

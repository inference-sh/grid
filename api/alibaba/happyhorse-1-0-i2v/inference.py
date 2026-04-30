"""
HappyHorse 1.0 Image-to-Video - Alibaba Cloud Video Generation

Generate physically realistic videos with smooth motion from a single image
and optional text description using the HappyHorse 1.0 I2V model via DashScope API.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta, VideoResolution
from pydantic import Field
from typing import Optional
from enum import Enum
import logging

from .wan_video_helper import generate_video, extract_video_url, extract_usage, download_video


class ResolutionEnum(str, Enum):
    hd = "720P"
    fhd = "1080P"


class AppInput(BaseAppInput):
    """Input schema for HappyHorse 1.0 Image-to-Video."""

    prompt: Optional[str] = Field(
        default=None,
        description="Text description of the video to generate. Up to 5000 non-Chinese characters or 2500 Chinese characters."
    )
    first_frame: File = Field(
        description="First frame image. Formats: JPEG, JPG, PNG, WEBP. Resolution: width and height at least 300px. Aspect ratio: 1:2.5 to 2.5:1. Up to 10MB."
    )
    resolution: ResolutionEnum = Field(
        default=ResolutionEnum.fhd,
        description="Video resolution: 720P or 1080P (default). Output aspect ratio follows the input image."
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
    """Output schema for HappyHorse 1.0 Image-to-Video."""

    video: File = Field(description="Generated video in MP4 format.")


class App(BaseApp):
    """HappyHorse 1.0 Image-to-Video generation."""

    async def setup(self, metadata):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.model = "happyhorse-1.0-i2v"
        self.logger.info("HappyHorse 1.0 I2V initialized")

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        self.logger.info(f"I2V: generating from first frame image")
        self.logger.info(f"Resolution: {input_data.resolution.value}, Duration: {input_data.duration}s")

        media = [{"type": "first_frame", "url": input_data.first_frame.uri}]

        input_payload = {"media": media}
        if input_data.prompt:
            input_payload["prompt"] = input_data.prompt

        parameters = {
            "resolution": input_data.resolution.value,
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

        resolution_map = {
            720: VideoResolution.VIDEO_RES720_P,
            1080: VideoResolution.VIDEO_RES1080_P,
        }

        self.logger.info(f"Video generated: {sr}P, {duration}s")

        return AppOutput(
            video=File(path=video_path),
            output_meta=OutputMeta(
                outputs=[
                    VideoMeta(
                        resolution=resolution_map.get(sr, VideoResolution.VIDEO_RES1080_P),
                        seconds=float(duration),
                        extra={"model": self.model},
                    )
                ]
            ),
        )

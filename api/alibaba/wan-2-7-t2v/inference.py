"""
Wan 2.7 Text-to-Video - Alibaba Cloud Video Generation

Generate videos from text prompts using the Wan 2.7 T2V model via DashScope API.
Supports 720P/1080P resolution, 2-15 second duration, and prompt rewriting.
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
    """Input schema for Wan 2.7 Text-to-Video."""

    prompt: str = Field(
        description="Text prompt describing the video to generate. Supports Chinese and English, up to 5000 characters."
    )
    negative_prompt: Optional[str] = Field(
        default=None,
        description="Content to exclude from the video. Up to 500 characters."
    )
    resolution: ResolutionEnum = Field(
        default=ResolutionEnum.fhd,
        description="Video resolution: 720P or 1080P (default)."
    )
    duration: int = Field(
        default=5,
        ge=2,
        le=15,
        description="Video duration in seconds (2-15)."
    )
    prompt_extend: bool = Field(
        default=True,
        description="Enable prompt rewriting via LLM for better results. Increases processing time."
    )
    watermark: bool = Field(
        default=False,
        description="Add 'AI Generated' watermark to bottom-right corner."
    )
    seed: Optional[int] = Field(
        default=None,
        ge=0,
        le=2147483647,
        description="Random seed for reproducibility (0-2147483647)."
    )


class AppOutput(BaseAppOutput):
    """Output schema for Wan 2.7 Text-to-Video."""

    video: File = Field(description="Generated video in MP4 format.")


class App(BaseApp):
    """Wan 2.7 Text-to-Video generation."""

    async def setup(self, metadata):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.model = "wan2.7-t2v"
        self.logger.info("Wan 2.7 T2V initialized")

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        self.logger.info(f"Generating video: {input_data.prompt[:100]}...")
        self.logger.info(f"Resolution: {input_data.resolution.value}, Duration: {input_data.duration}s")

        input_payload = {"prompt": input_data.prompt}
        if input_data.negative_prompt:
            input_payload["negative_prompt"] = input_data.negative_prompt

        parameters = {
            "resolution": input_data.resolution.value,
            "duration": input_data.duration,
            "prompt_extend": input_data.prompt_extend,
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

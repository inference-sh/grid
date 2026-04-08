"""
Wan 2.7 Image-to-Video - Alibaba Cloud Video Generation

Generate videos from images using the Wan 2.7 I2V model via DashScope API.
Supports three tasks:
- Video generation from first frame
- Video generation from first and last frames
- Video continuation from an existing clip
Also supports driving audio for lip-sync and action timing.
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
    """Input schema for Wan 2.7 Image-to-Video."""

    prompt: Optional[str] = Field(
        default=None,
        description="Text prompt describing video content. Supports Chinese and English, up to 5000 characters."
    )
    negative_prompt: Optional[str] = Field(
        default=None,
        description="Content to exclude from the video. Up to 500 characters."
    )
    first_frame: Optional[File] = Field(
        default=None,
        description="First frame image. Formats: JPEG, JPG, PNG, BMP, WEBP. Resolution: 240-8000px. Up to 20MB."
    )
    last_frame: Optional[File] = Field(
        default=None,
        description="Last frame image for first+last frame generation. Same format limits as first_frame."
    )
    driving_audio: Optional[File] = Field(
        default=None,
        description="Audio file for driving video generation (lip-sync, action timing). WAV/MP3, 2-30s, up to 15MB."
    )
    first_clip: Optional[File] = Field(
        default=None,
        description="Video clip for continuation. MP4/MOV, 2-10s, 240-4096px, up to 100MB."
    )
    resolution: ResolutionEnum = Field(
        default=ResolutionEnum.fhd,
        description="Video resolution: 720P or 1080P (default)."
    )
    duration: int = Field(
        default=5,
        ge=2,
        le=15,
        description="Video duration in seconds (2-15). For continuation, total output length including input clip."
    )
    prompt_extend: bool = Field(
        default=True,
        description="Enable prompt rewriting via LLM for better results."
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
    """Output schema for Wan 2.7 Image-to-Video."""

    video: File = Field(description="Generated video in MP4 format.")


class App(BaseApp):
    """Wan 2.7 Image-to-Video generation."""

    async def setup(self, metadata):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.model = "wan2.7-i2v"
        self.logger.info("Wan 2.7 I2V initialized")

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        # Build media array from provided inputs
        media = []
        if input_data.first_frame:
            media.append({"type": "first_frame", "url": input_data.first_frame.uri})
        if input_data.last_frame:
            media.append({"type": "last_frame", "url": input_data.last_frame.uri})
        if input_data.driving_audio:
            media.append({"type": "driving_audio", "url": input_data.driving_audio.uri})
        if input_data.first_clip:
            media.append({"type": "first_clip", "url": input_data.first_clip.uri})

        if not media:
            raise RuntimeError("At least one media input is required (first_frame, last_frame, driving_audio, or first_clip).")

        task_type = "continuation" if input_data.first_clip else "first+last frame" if input_data.last_frame else "first frame"
        self.logger.info(f"I2V mode: {task_type}")
        self.logger.info(f"Media assets: {len(media)}, Resolution: {input_data.resolution.value}, Duration: {input_data.duration}s")

        input_payload = {"media": media}
        if input_data.prompt:
            input_payload["prompt"] = input_data.prompt
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
                        extra={"model": self.model, "task_type": task_type},
                    )
                ]
            ),
        )

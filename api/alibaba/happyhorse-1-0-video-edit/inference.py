"""
HappyHorse 1.0 Video Edit - Alibaba Cloud Video Editing

Edit videos through natural language instructions with up to 5 reference images.
Supports local or global editing of video elements while preserving original
motion dynamics. Uses the HappyHorse 1.0 Video Edit model via DashScope API.
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


class AudioSettingEnum(str, Enum):
    auto = "auto"
    origin = "origin"


class AppInput(BaseAppInput):
    """Input schema for HappyHorse 1.0 Video Edit."""

    prompt: str = Field(
        description="Editing instruction describing what to change. Up to 5000 non-Chinese characters. E.g., 'Make the character wear the striped sweater from the image'."
    )
    video: File = Field(
        description="Video to edit. MP4/MOV (H.264 recommended). Duration: 3-60s. Resolution: longer side <= 2160px, shorter side >= 320px. Aspect ratio: 1:2.5 to 2.5:1. Up to 100MB. Frame rate > 8fps."
    )
    reference_images: Optional[List[File]] = Field(
        default=None,
        description="Reference images for editing (e.g., clothing to apply). Up to 5 images. JPEG/JPG/PNG/WEBP, width and height at least 300px. Up to 10MB each."
    )
    resolution: ResolutionEnum = Field(
        default=ResolutionEnum.fhd,
        description="Output resolution: 720P or 1080P (default)."
    )
    watermark: bool = Field(
        default=False,
        description="Add 'HappyHorse' watermark to bottom-right corner."
    )
    audio_setting: AudioSettingEnum = Field(
        default=AudioSettingEnum.auto,
        description="Audio handling: 'auto' (model decides) or 'origin' (keep original audio)."
    )
    seed: Optional[int] = Field(
        default=None,
        ge=0,
        le=2147483647,
        description="Random seed for reproducibility (0-2147483647)."
    )


class AppOutput(BaseAppOutput):
    """Output schema for HappyHorse 1.0 Video Edit."""

    video: File = Field(description="Edited video in MP4 format.")


class App(BaseApp):
    """HappyHorse 1.0 Video Edit application."""

    async def setup(self, metadata):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.model = "happyhorse-1.0-video-edit"
        self.logger.info("HappyHorse 1.0 Video Edit initialized")

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        media = [{"type": "video", "url": input_data.video.uri}]

        if input_data.reference_images:
            if len(input_data.reference_images) > 5:
                raise RuntimeError("Maximum 5 reference images allowed.")
            for img in input_data.reference_images:
                media.append({"type": "reference_image", "url": img.uri})

        num_refs = len(input_data.reference_images or [])
        edit_type = "instruction-only" if not num_refs else f"reference edit ({num_refs} images)"
        self.logger.info(f"VideoEdit mode: {edit_type}")
        self.logger.info(f"Resolution: {input_data.resolution.value}, Audio: {input_data.audio_setting.value}")

        input_payload = {
            "prompt": input_data.prompt,
            "media": media,
        }

        parameters = {
            "resolution": input_data.resolution.value,
            "watermark": input_data.watermark,
            "audio_setting": input_data.audio_setting.value,
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
        duration = usage.get("duration", 0)
        input_duration = usage.get("input_video_duration", 0)
        output_duration = usage.get("output_video_duration", 0)

        resolution_map = {
            720: VideoResolution.VIDEO_RES720_P,
            1080: VideoResolution.VIDEO_RES1080_P,
        }

        self.logger.info(f"Video edited: {sr}P, input={input_duration}s, output={output_duration}s, billed={duration}s")

        return AppOutput(
            video=File(path=video_path),
            output_meta=OutputMeta(
                outputs=[
                    VideoMeta(
                        resolution=resolution_map.get(sr, VideoResolution.VIDEO_RES1080_P),
                        seconds=float(duration) if duration else float(output_duration),
                        extra={
                            "model": self.model,
                            "edit_type": edit_type,
                            "input_video_duration": input_duration,
                            "output_video_duration": output_duration,
                        },
                    )
                ]
            ),
        )

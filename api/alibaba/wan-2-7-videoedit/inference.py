"""
Wan 2.7 Video Edit - Alibaba Cloud Video Editing

Edit videos using instruction-based prompts with the Wan 2.7 VideoEdit model
via DashScope API. Supports:
- Video style modification (e.g., claymation, anime)
- Object/clothing replacement using reference images
- Content editing via text instructions
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


class AudioSettingEnum(str, Enum):
    auto = "auto"
    origin = "origin"


class AppInput(BaseAppInput):
    """Input schema for Wan 2.7 Video Edit."""

    prompt: Optional[str] = Field(
        default=None,
        description="Editing instruction describing what to change. Up to 5000 characters. E.g., 'Convert the scene to claymation style' or 'Replace the clothes with the ones from the reference image'."
    )
    negative_prompt: Optional[str] = Field(
        default=None,
        description="Content to exclude from the video. Up to 500 characters."
    )
    video: File = Field(
        description="Video to edit. MP4/MOV, 2-10s, 240-4096px, up to 100MB."
    )
    reference_images: Optional[List[File]] = Field(
        default=None,
        description="Reference images for editing (e.g., clothing to apply). Up to 4 images. JPEG/JPG/PNG/BMP/WEBP, 240-8000px, up to 20MB each."
    )
    resolution: ResolutionEnum = Field(
        default=ResolutionEnum.fhd,
        description="Output resolution: 720P or 1080P (default)."
    )
    ratio: Optional[AspectRatioEnum] = Field(
        default=None,
        description="Output aspect ratio. If omitted, matches input video ratio."
    )
    duration: Optional[int] = Field(
        default=None,
        ge=2,
        le=10,
        description="Truncate video to this duration (seconds). Omit or 0 to keep original duration."
    )
    audio_setting: AudioSettingEnum = Field(
        default=AudioSettingEnum.auto,
        description="Audio handling: 'auto' (model decides based on prompt) or 'origin' (keep original audio)."
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
    """Output schema for Wan 2.7 Video Edit."""

    video: File = Field(description="Edited video in MP4 format.")


class App(BaseApp):
    """Wan 2.7 Video Edit application."""

    async def setup(self, metadata):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.model = "wan2.7-videoedit"
        self.logger.info("Wan 2.7 VideoEdit initialized")

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        # Build media array - video is required
        media = [{"type": "video", "url": input_data.video.uri}]

        if input_data.reference_images:
            for img in input_data.reference_images:
                media.append({"type": "reference_image", "url": img.uri})

        num_refs = len(input_data.reference_images or [])
        edit_type = "style transfer" if not num_refs else f"reference edit ({num_refs} images)"
        self.logger.info(f"VideoEdit mode: {edit_type}")
        self.logger.info(f"Resolution: {input_data.resolution.value}, Audio: {input_data.audio_setting.value}")

        input_payload = {"media": media}
        if input_data.prompt:
            input_payload["prompt"] = input_data.prompt
        if input_data.negative_prompt:
            input_payload["negative_prompt"] = input_data.negative_prompt

        parameters = {
            "resolution": input_data.resolution.value,
            "prompt_extend": input_data.prompt_extend,
            "watermark": input_data.watermark,
            "audio_setting": input_data.audio_setting.value,
        }
        if input_data.ratio:
            parameters["ratio"] = input_data.ratio.value
        if input_data.duration:
            parameters["duration"] = input_data.duration
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

"""
Wan 2.7 Reference-to-Video - Alibaba Cloud Video Generation

Generate videos featuring characters from reference images and videos using
the Wan 2.7 R2V model via DashScope API. Supports:
- Single or multi-character scenes from reference images/videos
- Voice timbre cloning from reference videos or audio
- First-frame control for precise scene composition
- Multi-panel storyboard input from a single image
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
    """Input schema for Wan 2.7 Reference-to-Video."""

    prompt: str = Field(
        description="Text prompt describing the video scene. Use 'Image 1', 'Image 2' to reference images and 'Video 1', 'Video 2' to reference videos in order of the media arrays. Up to 5000 characters."
    )
    negative_prompt: Optional[str] = Field(
        default=None,
        description="Content to exclude from the video. Up to 500 characters."
    )
    reference_images: Optional[List[File]] = Field(
        default=None,
        description="Reference images for characters/objects/scenes. Each must contain a single subject. Images + videos <= 5 total."
    )
    reference_videos: Optional[List[File]] = Field(
        default=None,
        description="Reference videos for characters and voice timbre. MP4/MOV, 1-30s, up to 100MB each. Avoid empty-scene videos."
    )
    first_frame: Optional[File] = Field(
        default=None,
        description="First frame image for precise scene composition. Max 1."
    )
    reference_voice: Optional[File] = Field(
        default=None,
        description="Audio for voice timbre reference. WAV/MP3, 1-10s, up to 15MB. Overrides reference video audio."
    )
    resolution: ResolutionEnum = Field(
        default=ResolutionEnum.fhd,
        description="Video resolution: 720P or 1080P (default)."
    )
    ratio: AspectRatioEnum = Field(
        default=AspectRatioEnum.landscape,
        description="Aspect ratio. Ignored if first_frame is provided (uses frame ratio instead)."
    )
    duration: int = Field(
        default=5,
        ge=2,
        le=15,
        description="Video duration in seconds. 2-10 if reference videos provided, 2-15 otherwise."
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
    """Output schema for Wan 2.7 Reference-to-Video."""

    video: File = Field(description="Generated video in MP4 format.")


class App(BaseApp):
    """Wan 2.7 Reference-to-Video generation."""

    async def setup(self, metadata):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.model = "wan2.7-r2v"
        self.logger.info("Wan 2.7 R2V initialized")

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        # Build media array
        media = []

        if input_data.reference_videos:
            for vid in input_data.reference_videos:
                entry = {"type": "reference_video", "url": vid.uri}
                media.append(entry)

        if input_data.reference_images:
            for img in input_data.reference_images:
                media.append({"type": "reference_image", "url": img.uri})

        if input_data.first_frame:
            media.append({"type": "first_frame", "url": input_data.first_frame.uri})

        if not media:
            raise RuntimeError("At least one reference image or video is required.")

        # Add reference_voice to the appropriate reference_video entries
        if input_data.reference_voice:
            for entry in media:
                if entry["type"] == "reference_video":
                    entry["reference_voice"] = input_data.reference_voice.uri
                    break
            else:
                # If no reference video, attach to first reference image
                for entry in media:
                    if entry["type"] == "reference_image":
                        entry["reference_voice"] = input_data.reference_voice.uri
                        break

        num_images = len(input_data.reference_images or [])
        num_videos = len(input_data.reference_videos or [])
        self.logger.info(f"R2V: {num_images} images, {num_videos} videos, first_frame={'yes' if input_data.first_frame else 'no'}")
        self.logger.info(f"Resolution: {input_data.resolution.value}, Ratio: {input_data.ratio.value}, Duration: {input_data.duration}s")

        input_payload = {
            "prompt": input_data.prompt,
            "media": media,
        }
        if input_data.negative_prompt:
            input_payload["negative_prompt"] = input_data.negative_prompt

        parameters = {
            "resolution": input_data.resolution.value,
            "ratio": input_data.ratio.value,
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
                        extra={"model": self.model, "ratio": ratio},
                    )
                ]
            ),
        )

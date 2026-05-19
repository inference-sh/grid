"""
HeyGen Avatar Video - Generate talking avatar videos.

Create videos with HeyGen's digital avatars speaking from a script,
with configurable voice, resolution, and aspect ratio.
"""

import logging
from typing import Optional, Literal

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta
from pydantic import Field

from .heygen_helper import (
    get_client,
    post_endpoint,
    poll_video,
    download_file,
)


ExpressivenessType = Literal["high", "medium", "low"]
ResolutionType = Literal["720p", "1080p", "4k"]
AspectRatioType = Literal["16:9", "9:16"]
OutputFormatType = Literal["mp4", "webm"]


class AppInput(BaseAppInput):
    """Input for avatar video generation."""

    avatar_id: str = Field(
        description="HeyGen avatar ID. Use the HeyGen dashboard or API to find available avatars."
    )
    script: str = Field(
        description="Text for the avatar to speak.",
        examples=["Hello! Welcome to our product demo. Let me show you how it works."],
    )
    voice_id: Optional[str] = Field(
        default=None,
        description="Voice ID for the avatar. If not set, the avatar's default voice is used.",
    )
    title: Optional[str] = Field(
        default=None,
        description="Title for the video.",
    )
    resolution: ResolutionType = Field(
        default="1080p",
        description="Video resolution.",
    )
    aspect_ratio: AspectRatioType = Field(
        default="16:9",
        description="Video aspect ratio.",
    )
    expressiveness: Optional[ExpressivenessType] = Field(
        default=None,
        description="Avatar expressiveness level. Only applies to photo avatars.",
    )
    motion_prompt: Optional[str] = Field(
        default=None,
        description="Natural language description of avatar motion. Only applies to photo avatars.",
    )
    remove_background: Optional[bool] = Field(
        default=None,
        description="Remove the video background.",
    )
    output_format: OutputFormatType = Field(
        default="mp4",
        description="Output video format.",
    )


class AppOutput(BaseAppOutput):
    """Output from avatar video generation."""

    video: File = Field(description="The generated avatar video.")


RESOLUTION_DIMENSIONS = {
    "720p": {"16:9": (1280, 720), "9:16": (720, 1280)},
    "1080p": {"16:9": (1920, 1080), "9:16": (1080, 1920)},
    "4k": {"16:9": (3840, 2160), "9:16": (2160, 3840)},
}


class App(BaseApp):
    async def setup(self):
        self.logger = logging.getLogger(__name__)

    async def run(self, input_data: AppInput) -> AppOutput:
        self.logger.info(f"Creating avatar video: {input_data.script[:100]}...")

        payload = {
            "type": "avatar",
            "avatar_id": input_data.avatar_id,
            "script": input_data.script,
            "aspect_ratio": input_data.aspect_ratio,
            "resolution": input_data.resolution,
            "output_format": input_data.output_format,
        }

        if input_data.voice_id:
            payload["voice_id"] = input_data.voice_id
        if input_data.title:
            payload["title"] = input_data.title
        if input_data.expressiveness:
            payload["expressiveness"] = input_data.expressiveness
        if input_data.motion_prompt:
            payload["motion_prompt"] = input_data.motion_prompt
        if input_data.remove_background is not None:
            payload["remove_background"] = input_data.remove_background

        async with get_client() as client:
            result = await post_endpoint(client, "/v3/videos", payload)
            video_id = result["video_id"]
            self.logger.info(f"Video created: {video_id}, polling...")

            completed = await poll_video(client, video_id)

        video_url = completed["video_url"]
        suffix = ".webm" if input_data.output_format == "webm" else ".mp4"
        video_path = await download_file(video_url, suffix=suffix)

        duration = completed.get("duration", 0)
        dims = RESOLUTION_DIMENSIONS.get(input_data.resolution, {}).get(
            input_data.aspect_ratio, (1920, 1080)
        )

        output_meta = OutputMeta(
            outputs=[
                VideoMeta(
                    width=dims[0],
                    height=dims[1],
                    seconds=float(duration),
                    fps=24,
                )
            ]
        )

        self.logger.info(f"Avatar video complete: {duration}s")
        return AppOutput(video=File(path=video_path), output_meta=output_meta)

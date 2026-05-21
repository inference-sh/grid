"""
HeyGen Photo Video - Animate still images into talking videos.

Upload a portrait image and generate a video of it speaking a script
with configurable voice, motion, and expressiveness.
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
    build_asset_ref,
)


ExpressivenessType = Literal["high", "medium", "low"]
ResolutionType = Literal["720p", "1080p", "4k"]
AspectRatioType = Literal["16:9", "9:16"]
OutputFormatType = Literal["mp4", "webm"]


class AppInput(BaseAppInput):
    """Input for photo video generation."""

    image: File = Field(
        description="Portrait image to animate. Should be a clear face photo."
    )
    script: Optional[str] = Field(
        default=None,
        description="Text for the animated image to speak.",
        examples=["Hi there! Thanks for checking out this demo."],
    )
    voice_id: Optional[str] = Field(
        default=None,
        description="Voice ID for speech. Required if script is provided.",
    )
    audio_url: Optional[str] = Field(
        default=None,
        description="URL to an audio file for the avatar to lip-sync to. Alternative to script+voice_id.",
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
    expressiveness: ExpressivenessType = Field(
        default="high",
        description="Expressiveness of the animated image.",
    )
    motion_prompt: Optional[str] = Field(
        default=None,
        description="Natural language description of desired motion (e.g., 'nodding slowly').",
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
    """Output from photo video generation."""

    video: File = Field(description="The generated video from the photo.")


RESOLUTION_DIMENSIONS = {
    "720p": {"16:9": (1280, 720), "9:16": (720, 1280)},
    "1080p": {"16:9": (1920, 1080), "9:16": (1080, 1920)},
    "4k": {"16:9": (3840, 2160), "9:16": (2160, 3840)},
}


class App(BaseApp):
    async def setup(self):
        self.logger = logging.getLogger(__name__)

    async def run(self, input_data: AppInput) -> AppOutput:
        self.logger.info("Creating photo video...")

        image_ref = build_asset_ref(input_data.image)

        payload = {
            "type": "image",
            "image": image_ref,
            "aspect_ratio": input_data.aspect_ratio,
            "resolution": input_data.resolution,
            "output_format": input_data.output_format,
            "expressiveness": input_data.expressiveness,
        }

        if input_data.script:
            payload["script"] = input_data.script
        if input_data.voice_id:
            payload["voice_id"] = input_data.voice_id
        if input_data.audio_url:
            payload["audio_url"] = input_data.audio_url
        if input_data.title:
            payload["title"] = input_data.title
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

        self.logger.info(f"Photo video complete: {duration}s")
        return AppOutput(video=File(path=video_path), output_meta=output_meta)

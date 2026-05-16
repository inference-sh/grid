"""
Kling Avatar - Digital Human Video from a Single Photo

Generate broadcast-style talking head videos from a single face photo.
Provide text+voice or audio to make the avatar speak.
"""

import os
import logging
from typing import Optional
from enum import Enum

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta
from pydantic import Field

from .kling_helper import (
    KlingClient,
    KlingAPIError,
    poll_task,
)
from .download_helper import download_video


class ModeEnum(str, Enum):
    std = "std"
    pro = "pro"


class AspectRatioEnum(str, Enum):
    r16_9 = "16:9"
    r9_16 = "9:16"
    r1_1 = "1:1"


class AppInput(BaseAppInput):
    """Kling Avatar - digital human from a photo.

    Provide a face image and an audio file for the avatar to speak.
    """

    image: File = Field(
        description="Face image for the avatar. Single clear front-facing face, good lighting. Formats: jpg, jpeg, png. Max 10MB.",
    )
    audio: Optional[File] = Field(
        default=None,
        description="Audio file for the avatar to speak (mp3/wav/m4a/aac, max 5MB, 2-300s).",
    )
    prompt: Optional[str] = Field(
        default=None,
        description="Action/emotion prompt for the avatar. E.g. 'While talking, excitedly shaking head'. Max 2500 chars.",
    )
    mode: ModeEnum = Field(
        default=ModeEnum.std,
        description="Quality mode. 'pro' for higher quality ($0.112/sec), 'std' for cheaper ($0.056/sec).",
    )
    aspect_ratio: AspectRatioEnum = Field(
        default=AspectRatioEnum.r16_9,
        description="Output video aspect ratio.",
    )


class AppOutput(BaseAppOutput):
    video: File = Field(description="The generated avatar video.")


class App(BaseApp):

    async def setup(self, metadata):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        logging.getLogger("httpx").setLevel(logging.WARNING)

        access_key = os.environ.get("KLING_ACCESS_KEY")
        secret_key = os.environ.get("KLING_SECRET_KEY")
        if not access_key or not secret_key:
            raise RuntimeError("KLING_ACCESS_KEY and KLING_SECRET_KEY must be set")

        self.client = KlingClient(access_key=access_key, secret_key=secret_key)
        self.logger.info("Kling Avatar initialized")

    async def on_cancel(self):
        self.logger.info("Cancellation requested")
        return True

    async def run(self, input_data: AppInput) -> AppOutput:
        if not input_data.audio:
            raise RuntimeError("Audio file is required for avatar generation")

        self.logger.info(f"Creating avatar video, quality: {input_data.mode.value}")

        task = await self.client.avatar.create(
            image=input_data.image.uri,
            sound_file=input_data.audio.uri,
            prompt=input_data.prompt,
            mode=input_data.mode.value,
        )

        self.logger.info(f"Task created: {task.task_id}")

        result = await poll_task(
            self.client.avatar.get,
            task.task_id,
            interval=3.0,
            timeout=600.0,
        )

        if not result.videos or not result.videos[0].url:
            raise RuntimeError(f"No video URL in result: {result.task_status_msg}")

        video_url = result.videos[0].url
        video_duration = float(result.videos[0].duration) if result.videos[0].duration else 10.0
        self.logger.info(f"Video ready: {video_url[:80]}..., duration={video_duration}s")

        video_path = download_video(video_url, self.logger)

        output_meta = OutputMeta(
            outputs=[
                VideoMeta(
                    seconds=video_duration,
                    extra={
                        "quality": input_data.mode.value,
                        "model": "kling-avatar",
                    },
                )
            ]
        )

        return AppOutput(video=File(path=video_path), output_meta=output_meta)

    async def unload(self):
        await self.client.close()

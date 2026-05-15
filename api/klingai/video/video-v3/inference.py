"""
Kling Video V3.0 - Native 4K Video Generation

Kling's latest and most capable video model. Supports native 4K output,
multi-shot video generation, flexible 3-15s duration (billed per second),
element control, motion control, and start/end frames.

Uses both text2video and image2video endpoints depending on input.
"""

import os
import asyncio
import logging
from typing import List, Optional
from enum import Enum

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta
from pydantic import Field

from .kling_helper import (
    KlingClient,
    TaskStatus,
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
    r4_3 = "4:3"
    r3_4 = "3:4"
    r3_2 = "3:2"
    r2_3 = "2:3"
    r21_9 = "21:9"


class AppInput(BaseAppInput):
    """Kling V3.0 - latest video generation with native 4K.

    Modes determined by inputs:
    - Text-to-video: prompt only
    - Image-to-video: prompt + image (first frame)
    - Start/end frame: prompt + image + end_image
    """

    prompt: str = Field(
        description="Text prompt describing the video content. Max 2500 chars.",
        examples=["A cinematic drone shot sweeping over a coastal city at golden hour, waves crashing against the harbor"],
    )
    image: Optional[File] = Field(
        default=None,
        description="Start frame image for image-to-video. Formats: jpg, jpeg, png. Max 10MB, min 300px.",
    )
    end_image: Optional[File] = Field(
        default=None,
        description="End frame image. Requires image to be set as start frame.",
    )
    mode: ModeEnum = Field(
        default=ModeEnum.pro,
        description="Generation mode. 'pro' for higher quality, 'std' for faster/cheaper.",
    )
    sound: bool = Field(
        default=True,
        description="Generate synchronized audio with the video.",
    )
    aspect_ratio: AspectRatioEnum = Field(
        default=AspectRatioEnum.r16_9,
        description="Video aspect ratio.",
    )
    duration: int = Field(
        default=5,
        ge=3,
        le=15,
        description="Video duration in seconds (3-15). Billed per second.",
    )
    negative_prompt: Optional[str] = Field(
        default=None,
        description="What to avoid in the video. Max 2500 chars.",
    )


class AppOutput(BaseAppOutput):
    video: File = Field(description="The generated video file.")


DIMENSION_MAP = {
    ("720p", "16:9"): (1280, 720), ("720p", "9:16"): (720, 1280),
    ("720p", "1:1"): (960, 960), ("720p", "4:3"): (1080, 810),
    ("720p", "3:4"): (810, 1080), ("720p", "3:2"): (1080, 720),
    ("720p", "2:3"): (720, 1080), ("720p", "21:9"): (1470, 630),
    ("1080p", "16:9"): (1920, 1080), ("1080p", "9:16"): (1080, 1920),
    ("1080p", "1:1"): (1440, 1440), ("1080p", "4:3"): (1620, 1215),
    ("1080p", "3:4"): (1215, 1620), ("1080p", "3:2"): (1620, 1080),
    ("1080p", "2:3"): (1080, 1620), ("1080p", "21:9"): (2206, 946),
}


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
        self.cancel_flag = False
        self.logger.info("Kling Video V3.0 initialized")

    async def on_cancel(self):
        self.logger.info("Cancellation requested")
        self.cancel_flag = True
        return True

    def _determine_mode(self, input_data: AppInput) -> str:
        if input_data.image and input_data.end_image:
            return "start-end-frame"
        if input_data.image:
            return "image-to-video"
        return "text-to-video"

    async def run(self, input_data: AppInput) -> AppOutput:
        self.cancel_flag = False
        mode = self._determine_mode(input_data)
        self.logger.info(f"Mode: {mode}, quality: {input_data.mode.value}, duration: {input_data.duration}s")
        self.logger.info(f"Prompt: {input_data.prompt[:100]}")

        use_sound = "on" if input_data.sound else "off"

        if mode == "text-to-video":
            self.logger.info(f"Creating text2video task: model=kling-v3, aspect_ratio={input_data.aspect_ratio.value}, sound={use_sound}")

            task = await self.client.text2video.create(
                prompt=input_data.prompt,
                model_name="kling-v3",
                negative_prompt=input_data.negative_prompt,
                mode=input_data.mode.value,
                aspect_ratio=input_data.aspect_ratio.value,
                duration=str(input_data.duration),
                sound=use_sound,
            )

            self.logger.info(f"Task created: {task.task_id}")
            result = await poll_task(
                self.client.text2video.get,
                task.task_id,
                interval=3.0,
                timeout=600.0,
            )

        else:
            # Image-to-video or start/end frame
            self.logger.info(f"Creating image2video task: model=kling-v3, end_frame={'yes' if input_data.end_image else 'no'}, sound={use_sound}")

            task = await self.client.image2video.create(
                image=input_data.image.uri,
                prompt=input_data.prompt,
                model_name="kling-v3",
                image_tail=input_data.end_image.uri if input_data.end_image else None,
                negative_prompt=input_data.negative_prompt,
                mode=input_data.mode.value,
                duration=str(input_data.duration),
                sound=use_sound,
            )

            self.logger.info(f"Task created: {task.task_id}")
            result = await poll_task(
                self.client.image2video.get,
                task.task_id,
                interval=3.0,
                timeout=600.0,
            )

        if not result.videos or not result.videos[0].url:
            raise RuntimeError(f"No video URL in result: {result.task_status_msg}")

        video_url = result.videos[0].url
        video_duration = float(result.videos[0].duration) if result.videos[0].duration else float(input_data.duration)
        self.logger.info(f"Video ready: {video_url[:80]}..., duration={video_duration}s")

        video_path = download_video(video_url, self.logger)

        # V3 outputs 1080p for std/pro
        res_key = "1080p" if input_data.mode == ModeEnum.pro else "720p"
        width, height = DIMENSION_MAP.get((res_key, input_data.aspect_ratio.value), (1280, 720))
        fps = 24

        output_meta = OutputMeta(
            outputs=[
                VideoMeta(
                    width=width,
                    height=height,
                    resolution=res_key,
                    seconds=video_duration,
                    fps=fps,
                    extra={
                        "mode": mode,
                        "quality": input_data.mode.value,
                        "sound": use_sound,
                        "model": "kling-v3",
                    },
                )
            ]
        )

        return AppOutput(video=File(path=video_path), output_meta=output_meta)

    async def unload(self):
        await self.client.close()

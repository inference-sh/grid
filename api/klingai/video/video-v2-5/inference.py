"""
Kling Video V2.5 Turbo - Fast Video Generation

Fast turbo model for text-to-video and image-to-video generation.
Supports start/end frames in pro mode. Optimized for speed while
maintaining good quality.
"""

import os
import asyncio
import logging
from typing import Optional
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


class DurationEnum(str, Enum):
    s5 = "5"
    s10 = "10"


class AppInput(BaseAppInput):
    """Kling V2.5 Turbo - fast video generation.

    Modes determined by inputs:
    - Text-to-video: prompt only
    - Image-to-video: prompt + image (first frame)
    - Start/end frame: prompt + image + end_image (pro mode only)
    """

    prompt: str = Field(
        description="Text prompt describing the video content. Max 2500 chars.",
        examples=["A majestic eagle soaring over snow-capped mountains at sunrise"],
    )
    image: Optional[File] = Field(
        default=None,
        description="Start frame image for image-to-video. Formats: jpg, jpeg, png. Max 10MB, min 300px.",
    )
    end_image: Optional[File] = Field(
        default=None,
        description="End frame image (pro mode only). Requires image to be set as start frame.",
    )
    mode: ModeEnum = Field(
        default=ModeEnum.pro,
        description="Generation mode. 'pro' for higher quality + start/end frame support, 'std' for faster/cheaper.",
    )
    aspect_ratio: AspectRatioEnum = Field(
        default=AspectRatioEnum.r16_9,
        description="Video aspect ratio.",
    )
    duration: DurationEnum = Field(
        default=DurationEnum.s5,
        description="Video duration: 5 or 10 seconds.",
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
        self.logger.info("Kling Video V2.5 Turbo initialized")

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
        self.logger.info(f"Mode: {mode}, quality: {input_data.mode.value}, duration: {input_data.duration.value}s")
        self.logger.info(f"Prompt: {input_data.prompt[:100]}")

        if mode == "start-end-frame" and input_data.mode != ModeEnum.pro:
            self.logger.warning("Start/end frame only supported in pro mode, upgrading to pro")

        if mode == "text-to-video":
            self.logger.info(f"Creating text2video task: aspect_ratio={input_data.aspect_ratio.value}")

            task = await self.client.text2video.create(
                prompt=input_data.prompt,
                model_name="kling-v2-5-turbo",
                negative_prompt=input_data.negative_prompt,
                mode=input_data.mode.value,
                aspect_ratio=input_data.aspect_ratio.value,
                duration=input_data.duration.value,
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
            i2v_mode = "pro" if mode == "start-end-frame" else input_data.mode.value

            self.logger.info(f"Creating image2video task: mode={i2v_mode}, end_frame={'yes' if input_data.end_image else 'no'}")

            task = await self.client.image2video.create(
                image=input_data.image.uri,
                prompt=input_data.prompt,
                model_name="kling-v2-5-turbo",
                image_tail=input_data.end_image.uri if input_data.end_image else None,
                negative_prompt=input_data.negative_prompt,
                mode=i2v_mode,
                duration=input_data.duration.value,
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
        video_duration = float(result.videos[0].duration) if result.videos[0].duration else float(input_data.duration.value)
        self.logger.info(f"Video ready: {video_url[:80]}..., duration={video_duration}s")

        video_path = download_video(video_url, self.logger)

        # V2.5 turbo: pro outputs 1080p, std outputs 720p
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
                        "model": "kling-v2-5-turbo",
                    },
                )
            ]
        )

        return AppOutput(video=File(path=video_path), output_meta=output_meta)

    async def unload(self):
        await self.client.close()

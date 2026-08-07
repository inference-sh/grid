"""
Kling Video V2.5 Turbo - Fast Video Generation

Fast turbo model for text-to-video and image-to-video generation.
Supports start/end frames. Optimized for speed while maintaining good quality.
"""

import os
import logging
from typing import Optional
from enum import Enum

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta
from pydantic import Field

from .kling_helper import KlingClient, KlingAPIError, poll_task_v2
from .download_helper import download_video


class AspectRatioEnum(str, Enum):
    r16_9 = "16:9"
    r9_16 = "9:16"
    r1_1 = "1:1"


class ResolutionEnum(str, Enum):
    r720p = "720p"
    r1080p = "1080p"


class DurationEnum(int, Enum):
    s5 = 5
    s10 = 10


class AppInput(BaseAppInput):
    """Kling V2.5 Turbo - fast video generation.

    Modes determined by inputs:
    - Text-to-video: prompt only
    - Image-to-video: prompt + image (first frame)
    - Start/end frame: prompt + image + end_image
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
        description="End frame image. Requires image to be set as start frame.",
    )
    resolution: ResolutionEnum = Field(
        default=ResolutionEnum.r720p,
        description="Video resolution.",
    )
    aspect_ratio: AspectRatioEnum = Field(
        default=AspectRatioEnum.r16_9,
        description="Video aspect ratio.",
    )
    duration: DurationEnum = Field(
        default=DurationEnum.s5,
        description="Video duration: 5 or 10 seconds.",
    )


class AppOutput(BaseAppOutput):
    video: File = Field(description="The generated video file.")


DIMENSION_MAP = {
    ("720p", "16:9"): (1280, 720), ("720p", "9:16"): (720, 1280), ("720p", "1:1"): (960, 960),
    ("1080p", "16:9"): (1920, 1080), ("1080p", "9:16"): (1080, 1920), ("1080p", "1:1"): (1440, 1440),
}


class App(BaseApp):

    async def setup(self, metadata):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        logging.getLogger("httpx").setLevel(logging.WARNING)

        api_key = os.environ.get("KLING_KEY")
        access_key = os.environ.get("KLING_ACCESS_KEY")
        secret_key = os.environ.get("KLING_SECRET_KEY")
        if api_key:
            self.client = KlingClient(api_key=api_key)
        elif access_key and secret_key:
            self.client = KlingClient(access_key=access_key, secret_key=secret_key)
        else:
            raise RuntimeError("Set KLING_KEY (V2) or KLING_ACCESS_KEY + KLING_SECRET_KEY (V1)")
        self.logger.info("Kling Video V2.5 Turbo initialized")

    async def on_cancel(self):
        return True

    async def run(self, input_data: AppInput) -> AppOutput:
        has_image = input_data.image is not None
        has_end = input_data.end_image is not None
        mode = "start-end-frame" if has_image and has_end else "image-to-video" if has_image else "text-to-video"
        self.logger.info(f"Mode: {mode}, res: {input_data.resolution.value}, duration: {input_data.duration.value}s")

        settings = {
            "resolution": input_data.resolution.value,
            "aspect_ratio": input_data.aspect_ratio.value,
            "duration": input_data.duration.value,
        }

        if mode == "text-to-video":
            task = await self.client.v2.text_to_video(
                model="kling-2.5-turbo",
                prompt=input_data.prompt,
                settings=settings,
            )
        else:
            contents = [{"type": "prompt", "text": input_data.prompt}]
            contents.append({"type": "first_frame", "url": input_data.image.uri})
            if has_end:
                contents.append({"type": "last_frame", "url": input_data.end_image.uri})

            task = await self.client.v2.image_to_video(
                model="kling-2.5-turbo",
                contents=contents,
                settings=settings,
            )

        self.logger.info(f"Task created: {task.id}")
        result = await poll_task_v2(self.client, task.id, interval=3.0)

        video_out = next((o for o in (result.outputs or []) if o.type == "video"), None)
        if not video_out or not video_out.url:
            raise RuntimeError(f"No video URL: {result.message}")

        video_duration = float(video_out.duration) if video_out.duration else float(input_data.duration.value)
        video_path = download_video(video_out.url, self.logger)

        width, height = DIMENSION_MAP.get(
            (input_data.resolution.value, input_data.aspect_ratio.value), (1280, 720)
        )

        output_meta = OutputMeta(
            outputs=[
                VideoMeta(
                    width=width, height=height,
                    resolution=input_data.resolution.value,
                    seconds=video_duration, fps=24,
                    extra={"mode": mode, "model": "kling-2.5-turbo"},
                )
            ]
        )
        return AppOutput(video=File(path=video_path), output_meta=output_meta)

    async def unload(self):
        await self.client.close()

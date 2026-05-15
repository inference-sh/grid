"""
Kling Video V2.6 - Video Generation with Sound & Voice

Latest Kling model with native audio generation and voice control.
Supports text-to-video and image-to-video with start/end frames,
sound generation (pro mode), and voice-driven animation.
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


class DurationEnum(str, Enum):
    s5 = "5"
    s10 = "10"


class AppInput(BaseAppInput):
    """Kling V2.6 video generation with sound and voice support.

    Modes determined by inputs:
    - Text-to-video: prompt only
    - Image-to-video: prompt + image (first frame)
    - Start/end frame: prompt + image + end_image (pro only, no audio)
    - Voice control: prompt with <<<voice_1>>> + image + voice_id (pro only)
    """

    prompt: str = Field(
        description="Text prompt describing the video content. For voice control, include <<<voice_1>>> where the character speaks. Max 2500 chars.",
        examples=["Ocean waves crashing on rocky shore at golden hour"],
    )
    image: Optional[File] = Field(
        default=None,
        description="Start frame image for image-to-video. Formats: jpg, jpeg, png. Max 10MB, min 300px.",
    )
    end_image: Optional[File] = Field(
        default=None,
        description="End frame image (pro mode only, disables audio). Requires image to be set.",
    )
    voice_id: Optional[str] = Field(
        default=None,
        description="Voice ID for voice-driven animation (pro mode only). Use <<<voice_1>>> in prompt to mark speech.",
    )
    mode: ModeEnum = Field(
        default=ModeEnum.pro,
        description="Generation mode. 'pro' for higher quality + sound/voice support, 'std' for faster/cheaper (no audio in std).",
    )
    sound: bool = Field(
        default=True,
        description="Generate synchronized audio. Only works in pro mode. Disabled when using start/end frames.",
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
        description="What to avoid in the video. Max 2500 chars. Note: not supported on V2.x models.",
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
        self.logger.info("Kling Video V2.6 initialized")

    async def on_cancel(self):
        self.logger.info("Cancellation requested")
        self.cancel_flag = True
        return True

    def _determine_mode(self, input_data: AppInput) -> str:
        if input_data.voice_id and input_data.image:
            return "voice-control"
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

        # Determine sound setting
        use_sound = "off"
        if input_data.sound and input_data.mode == ModeEnum.pro:
            if mode == "start-end-frame":
                self.logger.info("Sound disabled for start/end frame mode")
            else:
                use_sound = "on"

        if mode == "text-to-video":
            self.logger.info(f"Creating text2video task: aspect_ratio={input_data.aspect_ratio.value}, sound={use_sound}")

            task = await self.client.text2video.create(
                prompt=input_data.prompt,
                model_name="kling-v2-6",
                mode=input_data.mode.value,
                aspect_ratio=input_data.aspect_ratio.value,
                duration=input_data.duration.value,
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
            # Image-to-video, start/end frame, or voice control
            voice_list = None
            if mode == "voice-control" and input_data.voice_id:
                from .kling_helper import VoiceRef
                voice_list = [VoiceRef(voice_id=input_data.voice_id)]
                use_sound = "on"

            self.logger.info(f"Creating image2video task: sound={use_sound}, voice={'yes' if voice_list else 'no'}")

            task = await self.client.image2video.create(
                image=input_data.image.uri,
                prompt=input_data.prompt,
                model_name="kling-v2-6",
                image_tail=input_data.end_image.uri if input_data.end_image else None,
                voice_list=voice_list,
                sound=use_sound,
                mode=input_data.mode.value,
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

        # V2.6 pro outputs 1080p, std outputs 720p
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
                        "model": "kling-v2-6",
                    },
                )
            ]
        )

        return AppOutput(video=File(path=video_path), output_meta=output_meta)

    async def unload(self):
        await self.client.close()

"""
Kling Video V2.6 - Video Generation with Sound & Voice

Kling model with native audio generation and voice control.
Supports text-to-video and image-to-video with start/end frames,
sound generation, and voice-driven animation.
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
    """Kling V2.6 video generation with sound and voice support.

    Modes determined by inputs:
    - Text-to-video: prompt only
    - Image-to-video: prompt + image (first frame)
    - Start/end frame: prompt + image + end_image
    - Voice control: prompt + image + voice_id
    """

    prompt: str = Field(
        description="Text prompt describing the video content. For voice control, include @voice_name where the character speaks. Max 2500 chars.",
        examples=["Ocean waves crashing on rocky shore at golden hour"],
    )
    image: Optional[File] = Field(
        default=None,
        description="Start frame image for image-to-video. Formats: jpg, jpeg, png. Max 10MB, min 300px.",
    )
    end_image: Optional[File] = Field(
        default=None,
        description="End frame image. Requires image to be set.",
    )
    voice_id: Optional[str] = Field(
        default=None,
        description="Voice ID for voice-driven animation. Use @voice_name in prompt to mark speech.",
    )
    sound: bool = Field(
        default=True,
        description="Generate synchronized native audio.",
    )
    resolution: ResolutionEnum = Field(
        default=ResolutionEnum.r1080p,
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
        self.logger.info("Kling Video V2.6 initialized")

    async def on_cancel(self):
        return True

    async def run(self, input_data: AppInput) -> AppOutput:
        has_image = input_data.image is not None
        has_end = input_data.end_image is not None
        has_voice = input_data.voice_id is not None

        if has_voice and has_image:
            mode = "voice-control"
        elif has_image and has_end:
            mode = "start-end-frame"
        elif has_image:
            mode = "image-to-video"
        else:
            mode = "text-to-video"

        self.logger.info(f"Mode: {mode}, res: {input_data.resolution.value}, duration: {input_data.duration.value}s")

        use_audio = "native" if input_data.sound else "off"
        if mode == "start-end-frame":
            use_audio = "off"

        settings = {
            "resolution": input_data.resolution.value,
            "duration": input_data.duration.value,
            "audio": use_audio,
        }

        if mode == "text-to-video":
            settings["aspect_ratio"] = input_data.aspect_ratio.value
            task = await self.client.v2.text_to_video(
                model="kling-2.6",
                prompt=input_data.prompt,
                settings=settings,
            )
        else:
            contents = [{"type": "prompt", "text": input_data.prompt}]
            contents.append({"type": "first_frame", "url": input_data.image.uri})
            if has_end:
                contents.append({"type": "last_frame", "url": input_data.end_image.uri})
            if has_voice:
                contents.append({"type": "voice", "voice_id": input_data.voice_id, "id": "voice_1"})
                use_audio = "native"
                settings["audio"] = use_audio

            task = await self.client.v2.image_to_video(
                model="kling-2.6",
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
            (input_data.resolution.value, input_data.aspect_ratio.value), (1920, 1080)
        )

        output_meta = OutputMeta(
            outputs=[
                VideoMeta(
                    width=width, height=height,
                    resolution=input_data.resolution.value,
                    seconds=video_duration, fps=24,
                    extra={"mode": mode, "model": "kling-2.6", "sound": use_audio},
                )
            ]
        )
        return AppOutput(video=File(path=video_path), output_meta=output_meta)

    async def unload(self):
        await self.client.close()

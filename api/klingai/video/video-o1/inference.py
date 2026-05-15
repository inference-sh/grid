"""
Kling Video O1 - Omni Video Generation

Unified video generation using Kling's most capable model (kling-video-o1).
Supports text-to-video, image-to-video with start/end frames, image/element
references, and video references for editing and style transfer.
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
    ImageRef,
    VideoRef,
    WatermarkInfo,
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


class AppInput(BaseAppInput):
    """Kling Video O1 (Omni) - unified video generation with references.

    Modes determined by inputs:
    - Text-to-video: prompt only (aspect_ratio required)
    - Image-to-video: prompt + image (first frame)
    - Start/end frame: prompt + image + end_image
    - Reference generation: prompt + reference_images / reference_video
    """

    prompt: str = Field(
        description="Text prompt describing the video. Use <<<image_1>>>, <<<element_1>>>, <<<video_1>>> to reference inputs. Max 2500 chars.",
        examples=["A serene lake at sunset with birds flying overhead"],
    )
    image: Optional[File] = Field(
        default=None,
        description="First-frame reference image. Sets the opening frame of the video.",
    )
    end_image: Optional[File] = Field(
        default=None,
        description="End-frame reference image. Requires image (first frame) to be set.",
    )
    reference_images: List[File] = Field(
        default=[],
        max_length=7,
        description="Reference images for style, character, or scene consistency. Referenced in prompt as <<<image_1>>>, <<<image_2>>>, etc. Max 7 without video, max 4 with video.",
    )
    reference_video: Optional[File] = Field(
        default=None,
        description="Reference video for camera style, motion, or editing. Referenced in prompt as <<<video_1>>>.",
    )
    reference_video_type: str = Field(
        default="feature",
        description="How to use reference video: 'feature' for style/motion reference, 'base' for direct editing/transformation.",
    )
    keep_original_sound: bool = Field(
        default=True,
        description="Keep original sound from reference video (only applies when reference_video is set).",
    )
    mode: ModeEnum = Field(
        default=ModeEnum.pro,
        description="Generation quality. 'pro' for highest quality, 'std' for faster/cheaper.",
    )
    aspect_ratio: Optional[AspectRatioEnum] = Field(
        default=None,
        description="Video aspect ratio. Required for text-to-video and reference generation (not needed when using first-frame image or video editing).",
    )
    duration: int = Field(
        default=5,
        ge=3,
        le=10,
        description="Video duration in seconds (3-10). For video editing, output matches input video duration.",
    )
    watermark: bool = Field(
        default=False,
        description="Add watermark to the output video.",
    )


class AppOutput(BaseAppOutput):
    video: File = Field(description="The generated video file.")


DIMENSION_MAP = {
    ("720p", "16:9"): (1280, 720),
    ("720p", "9:16"): (720, 1280),
    ("720p", "1:1"): (960, 960),
    ("1080p", "16:9"): (1920, 1080),
    ("1080p", "9:16"): (1080, 1920),
    ("1080p", "1:1"): (1440, 1440),
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
        self.logger.info("Kling Video O1 initialized")

    async def on_cancel(self):
        self.logger.info("Cancellation requested")
        self.cancel_flag = True
        return True

    def _determine_mode(self, input_data: AppInput) -> str:
        if input_data.reference_video and input_data.reference_video_type == "base":
            return "video-editing"
        if input_data.reference_video or input_data.reference_images:
            return "reference-generation"
        if input_data.image and input_data.end_image:
            return "start-end-frame"
        if input_data.image:
            return "image-to-video"
        return "text-to-video"

    async def run(self, input_data: AppInput) -> AppOutput:
        self.cancel_flag = False
        mode = self._determine_mode(input_data)
        self.logger.info(f"Mode: {mode}, prompt: {input_data.prompt[:100]}")

        # Build image_list
        image_list = []
        if input_data.image:
            image_list.append(ImageRef(image_url=input_data.image.uri, type="first_frame"))
        if input_data.end_image:
            if not input_data.image:
                raise RuntimeError("End frame requires a first frame image")
            image_list.append(ImageRef(image_url=input_data.end_image.uri, type="end_frame"))
        for ref_img in input_data.reference_images:
            image_list.append(ImageRef(image_url=ref_img.uri))

        # Build video_list
        video_list = []
        if input_data.reference_video:
            video_list.append(VideoRef(
                video_url=input_data.reference_video.uri,
                refer_type=input_data.reference_video_type,
                keep_original_sound="yes" if input_data.keep_original_sound else "no",
            ))

        # Determine if aspect_ratio is required
        needs_aspect_ratio = mode in ("text-to-video", "reference-generation")
        aspect_ratio = input_data.aspect_ratio
        if needs_aspect_ratio and not aspect_ratio:
            aspect_ratio = AspectRatioEnum.r16_9
            self.logger.info(f"Auto-selecting aspect_ratio: {aspect_ratio.value}")

        self.logger.info(f"Creating omni-video task: mode={input_data.mode.value}, duration={input_data.duration}s, images={len(image_list)}, videos={len(video_list)}")

        task = await self.client.omni_video.create(
            prompt=input_data.prompt,
            model_name="kling-video-o1",
            image_list=image_list if image_list else None,
            video_list=video_list if video_list else None,
            mode=input_data.mode.value,
            aspect_ratio=aspect_ratio.value if aspect_ratio else None,
            duration=str(input_data.duration),
            watermark_info=WatermarkInfo(enabled=input_data.watermark),
        )

        self.logger.info(f"Task created: {task.task_id}")

        # Poll for completion
        result = await poll_task(
            self.client.omni_video.get,
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

        # Build output metadata
        ratio_str = aspect_ratio.value if aspect_ratio else "16:9"
        # O1 outputs at various resolutions depending on mode
        res_key = "1080p" if input_data.mode == ModeEnum.pro else "720p"
        width, height = DIMENSION_MAP.get((res_key, ratio_str), (1280, 720))
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
                        "model": "kling-video-o1",
                    },
                )
            ]
        )

        return AppOutput(video=File(path=video_path), output_meta=output_meta)

    async def unload(self):
        await self.client.close()

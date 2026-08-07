"""
Kling Video O1 - Omni Video Generation

Unified video generation using Kling's most capable model (kling-o1).
Supports text-to-video, image-to-video with start/end frames, image/element
references, and video references for editing and style transfer.
"""

import os
import logging
from typing import List, Optional
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


class AppInput(BaseAppInput):
    """Kling Video O1 (Omni) - unified video generation with references.

    Modes determined by inputs:
    - Text-to-video: prompt only (aspect_ratio required)
    - Image-to-video: prompt + image (first frame)
    - Start/end frame: prompt + image + end_image
    - Reference generation: prompt + reference_images / reference_video
    """

    prompt: str = Field(
        description="Text prompt describing the video. Use @image_1, @element_name, @video_1 to reference inputs. Max 2500 chars.",
        examples=["A serene lake at sunset with birds flying overhead"],
    )
    image: Optional[File] = Field(
        default=None,
        description="First-frame reference image.",
    )
    end_image: Optional[File] = Field(
        default=None,
        description="End-frame reference image. Requires image (first frame) to be set.",
    )
    reference_images: List[File] = Field(
        default=[],
        max_length=7,
        description="Reference images for style, character, or scene consistency. Referenced in prompt as @image_1, @image_2, etc. Max 7 without video, max 4 with video.",
    )
    reference_video: Optional[File] = Field(
        default=None,
        description="Reference video for camera style, motion, or editing. Referenced in prompt as @video_1.",
    )
    reference_video_type: str = Field(
        default="feature",
        description="How to use reference video: 'feature' for style/motion reference, 'base' for direct editing.",
    )
    resolution: ResolutionEnum = Field(
        default=ResolutionEnum.r1080p,
        description="Video resolution.",
    )
    aspect_ratio: Optional[AspectRatioEnum] = Field(
        default=None,
        description="Video aspect ratio. Required for text-to-video.",
    )
    duration: int = Field(
        default=5,
        ge=3,
        le=10,
        description="Video duration in seconds. Text-to-video: 5 or 10 only. With reference images: 3-10.",
    )
    watermark: bool = Field(
        default=False,
        description="Add watermark to the output video.",
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
        self.logger.info("Kling Video O1 initialized")

    async def on_cancel(self):
        return True

    async def run(self, input_data: AppInput) -> AppOutput:
        has_ref_video = input_data.reference_video is not None
        has_ref_imgs = len(input_data.reference_images) > 0
        has_image = input_data.image is not None
        has_end = input_data.end_image is not None

        if has_ref_video and input_data.reference_video_type == "base":
            mode = "video-editing"
        elif has_ref_video or has_ref_imgs:
            mode = "reference-generation"
        elif has_image and has_end:
            mode = "start-end-frame"
        elif has_image:
            mode = "image-to-video"
        else:
            mode = "text-to-video"

        self.logger.info(f"Mode: {mode}, prompt: {input_data.prompt[:100]}")

        contents = [{"type": "prompt", "text": input_data.prompt}]

        if has_image:
            contents.append({"type": "first_frame", "url": input_data.image.uri})
        if has_end:
            if not has_image:
                raise RuntimeError("End frame requires a first frame image")
            contents.append({"type": "last_frame", "url": input_data.end_image.uri})

        for i, ref_img in enumerate(input_data.reference_images):
            contents.append({"type": "refer_image", "url": ref_img.uri, "id": f"image_{i+1}"})

        if has_ref_video:
            vid_type = "base_video" if input_data.reference_video_type == "base" else "feature_video"
            contents.append({"type": vid_type, "url": input_data.reference_video.uri, "id": "video_1"})

        aspect_ratio = input_data.aspect_ratio
        if mode in ("text-to-video", "reference-generation") and not aspect_ratio:
            aspect_ratio = AspectRatioEnum.r16_9

        settings = {
            "resolution": input_data.resolution.value,
            "duration": input_data.duration,
        }
        if aspect_ratio:
            settings["aspect_ratio"] = aspect_ratio.value

        options = {}
        if input_data.watermark:
            options["watermark_info"] = {"enabled": True}

        self.logger.info(f"Creating omni-video task: duration={input_data.duration}s, refs={len(contents)-1}")

        task = await self.client.v2.omni_video(
            model="kling-o1",
            contents=contents,
            settings=settings,
            options=options if options else None,
        )

        self.logger.info(f"Task created: {task.id}")
        result = await poll_task_v2(self.client, task.id, interval=3.0)

        video_out = next((o for o in (result.outputs or []) if o.type == "video"), None)
        if not video_out or not video_out.url:
            raise RuntimeError(f"No video URL: {result.message}")

        video_duration = float(video_out.duration) if video_out.duration else float(input_data.duration)
        video_path = download_video(video_out.url, self.logger)

        ratio_str = aspect_ratio.value if aspect_ratio else "16:9"
        width, height = DIMENSION_MAP.get((input_data.resolution.value, ratio_str), (1920, 1080))

        output_meta = OutputMeta(
            outputs=[
                VideoMeta(
                    width=width, height=height,
                    resolution=input_data.resolution.value,
                    seconds=video_duration, fps=24,
                    extra={"mode": mode, "model": "kling-o1"},
                )
            ]
        )
        return AppOutput(video=File(path=video_path), output_meta=output_meta)

    async def unload(self):
        await self.client.close()

"""
MiniMax H3 Max 3D to Video via fal.ai

Turn a Blender or 3D previs render into photorealistic video. Layout, camera movement and
timing follow the input video; optional reference images set the appearance.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta, ImageMeta
from pydantic import Field
from typing import Optional, List
from enum import Enum
import logging
import math

from .fal_helper import setup_fal_client, run_fal_model, download_video
from .h3max_common import dims_for, uris, probe_video, FPS

logging.getLogger("httpx").setLevel(logging.WARNING)

ENDPOINT = "minimax/h3-max/3d-to-video"


class ResolutionEnum(str, Enum):
    r480p = "480P"
    r768p = "768P"
    r1080p = "1080P"


class AppInput(BaseAppInput):
    video: File = Field(description="Blender or previs render, up to 15 seconds and 32 shots.")
    prompt: Optional[str] = Field(
        default=None,
        description="Optional clarification of what the proxies represent or how they move, e.g. 'The moving block is a running person.'",
    )
    reference_images: Optional[List[File]] = Field(
        default=None, max_length=8,
        description="Optional environment, subject, interior or detail references (max 8). Used directly, without generating new images.",
    )
    max_generated_reference_images: int = Field(
        default=2, ge=1, le=8,
        description="Max new appearance references to generate when no reference images are given. Each is billed by fal.",
    )
    resolution: ResolutionEnum = Field(default=ResolutionEnum.r768p, description="Output quality. Duration follows the input video.")


class AppOutput(BaseAppOutput):
    video: File = Field(description="The finished realistic video, matching the input duration.")


class App(BaseApp):

    async def setup(self, metadata):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        self.logger.info("MiniMax H3 Max 3D to Video initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        setup_fal_client()
        request = {
            "video_url": input_data.video.uri,
            "resolution": input_data.resolution.value,
            "max_generated_reference_images": input_data.max_generated_reference_images,
        }
        if input_data.prompt:
            request["prompt"] = input_data.prompt
        if input_data.reference_images:
            request["reference_image_urls"] = uris(input_data.reference_images)
        self.logger.info(f"3D to video: {input_data.resolution.value}, references={len(input_data.reference_images or [])}, "
                         f"max_generated={input_data.max_generated_reference_images}, endpoint: {ENDPOINT}")
        result = run_fal_model(ENDPOINT, request, self.logger)
        video_path = download_video(result["video"]["url"], self.logger)

        info = probe_video(video_path, self.logger)
        width, height = (info.get("width"), info.get("height")) if info.get("width") else dims_for(input_data.resolution.value, "16:9")
        seconds = float(info.get("seconds") or 0)
        # fal bills whole seconds per shot with a 5s minimum per shot; shots are not reported, so bill the rounded-up total with a 5s floor.
        billed_seconds = float(max(5, math.ceil(seconds))) if seconds else 5.0
        generated_refs = 0 if input_data.reference_images else input_data.max_generated_reference_images
        output_meta = OutputMeta(
            inputs=[VideoMeta()] + [ImageMeta() for _ in (input_data.reference_images or [])],
            outputs=[VideoMeta(width=int(width), height=int(height), resolution=input_data.resolution.value,
                               seconds=billed_seconds, fps=int(info.get("fps") or FPS),
                               extra={"actual_seconds": seconds, "generated_reference_images": generated_refs})],
        )
        self.logger.info(f"Generated {width}x{height} video, {seconds}s (billed {billed_seconds}s, {generated_refs} generated refs)")
        return AppOutput(video=File(path=video_path), output_meta=output_meta)

"""
MiniMax H3 Max Relight via fal.ai

Relight an existing video from a lighting sphere reference image. Subjects, motion, camera
and audio are kept; only the lighting changes.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta, ImageMeta
from pydantic import Field
from typing import Optional
from enum import Enum
import logging

from .fal_helper import setup_fal_client, run_fal_model, download_video
from .h3max_common import dims_for, probe_video, FPS

logging.getLogger("httpx").setLevel(logging.WARNING)

ENDPOINT = "minimax/h3-max/relight"


class ResolutionEnum(str, Enum):
    r480p = "480P"
    r768p = "768P"
    r1080p = "1080P"
    r2k = "2K"


class AspectRatioEnum(str, Enum):
    adaptive = "adaptive"
    r21_9 = "21:9"
    r16_9 = "16:9"
    r4_3 = "4:3"
    r1_1 = "1:1"
    r3_4 = "3:4"
    r9_16 = "9:16"


class AppInput(BaseAppInput):
    video: File = Field(description="Source video up to 15 seconds (mp4, mov, webm, m4v, gif). Trimmed to a 17n+5 frame count at 24 fps.")
    reference_image: File = Field(description="Lighting reference: a sphere render showing the target light positions, colors and intensity.")
    resolution: ResolutionEnum = Field(default=ResolutionEnum.r768p, description="480P and 768P native; 1080P and 2K refined from 768P.")
    aspect_ratio: AspectRatioEnum = Field(default=AspectRatioEnum.adaptive, description="Output aspect ratio. adaptive follows the source.")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducible generation.")


class AppOutput(BaseAppOutput):
    video: File = Field(description="The relighted video.")
    seed: Optional[int] = Field(default=None, description="Seed used for generation.")


class App(BaseApp):

    async def setup(self, metadata):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        self.logger.info("MiniMax H3 Max Relight initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        setup_fal_client()
        request = {
            "video_url": input_data.video.uri,
            "reference_image_url": input_data.reference_image.uri,
            "resolution": input_data.resolution.value,
            "aspect_ratio": input_data.aspect_ratio.value,
        }
        if input_data.seed is not None:
            request["seed"] = input_data.seed
        self.logger.info(f"Relight: {input_data.resolution.value}, {input_data.aspect_ratio.value}, endpoint: {ENDPOINT}")
        result = run_fal_model(ENDPOINT, request, self.logger)
        self.logger.info(f"fal timings: {result.get('timings')}")
        video_path = download_video(result["video"]["url"], self.logger)

        info = probe_video(video_path, self.logger)
        width, height = (info.get("width"), info.get("height")) if info.get("width") else dims_for(input_data.resolution.value, input_data.aspect_ratio.value)
        seconds = float(info.get("seconds") or 0)
        output_meta = OutputMeta(
            inputs=[VideoMeta(), ImageMeta()],
            outputs=[VideoMeta(width=int(width), height=int(height), resolution=input_data.resolution.value,
                               seconds=seconds, fps=int(info.get("fps") or FPS))],
        )
        self.logger.info(f"Relighted {width}x{height} video, {seconds}s")
        return AppOutput(video=File(path=video_path), seed=result.get("seed"), output_meta=output_meta)

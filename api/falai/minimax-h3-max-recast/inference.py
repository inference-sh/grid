"""
MiniMax H3 Max Recast via fal.ai

Replace the people in a video with the people in reference photos. Motion, camera work,
cuts and audio are kept.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta, ImageMeta
from pydantic import Field
from typing import Optional, List
from enum import Enum
import logging

from .fal_helper import setup_fal_client, run_fal_model, download_video
from .h3max_common import dims_for, uris, probe_video, FPS

logging.getLogger("httpx").setLevel(logging.WARNING)

ENDPOINT = "minimax/h3-max/recast"


class ResolutionEnum(str, Enum):
    r768p = "768P"
    r1080p = "1080P"


class AppInput(BaseAppInput):
    video: File = Field(description="Source video, 5 to 30 seconds, with no single shot longer than 15 seconds.")
    reference_images: List[File] = Field(
        min_length=1, max_length=4,
        description="One photo per new person (max 4). By default they replace the main people from left to right, in every shot they appear in.",
    )
    prompt: Optional[str] = Field(default=None, description="Optional: who becomes whom, or what else to keep or change.")
    resolution: ResolutionEnum = Field(default=ResolutionEnum.r1080p, description="Output resolution.")
    enhance_realism: bool = Field(default=False, description="More natural skin textures and better blending of the new people into the scene.")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducible generation.")


class AppOutput(BaseAppOutput):
    video: File = Field(description="The recast video, with the source sound.")
    seed: Optional[int] = Field(default=None, description="Seed used for generation.")


class App(BaseApp):

    async def setup(self, metadata):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        self.logger.info("MiniMax H3 Max Recast initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        setup_fal_client()
        request = {
            "video_url": input_data.video.uri,
            "reference_image_urls": uris(input_data.reference_images),
            "resolution": input_data.resolution.value,
            "enhance_realism": input_data.enhance_realism,
        }
        if input_data.prompt:
            request["prompt"] = input_data.prompt
        if input_data.seed is not None:
            request["seed"] = input_data.seed
        self.logger.info(f"Recast: {len(input_data.reference_images)} people, {input_data.resolution.value}, "
                         f"enhance_realism={input_data.enhance_realism}, endpoint: {ENDPOINT}")
        result = run_fal_model(ENDPOINT, request, self.logger)
        video_path = download_video(result["video"]["url"], self.logger)

        info = probe_video(video_path, self.logger)
        width, height = (info.get("width"), info.get("height")) if info.get("width") else dims_for(input_data.resolution.value, "16:9")
        seconds = float(info.get("seconds") or 0)
        output_meta = OutputMeta(
            inputs=[VideoMeta()] + [ImageMeta() for _ in input_data.reference_images],
            outputs=[VideoMeta(width=int(width), height=int(height), resolution=input_data.resolution.value,
                               seconds=seconds, fps=int(info.get("fps") or FPS),
                               extra={"enhance_realism": input_data.enhance_realism})],
        )
        self.logger.info(f"Recast {width}x{height} video, {seconds}s")
        return AppOutput(video=File(path=video_path), seed=result.get("seed"), output_meta=output_meta)

"""
VACE - VACE-14B video generation and editing by Pruna
"""

import logging
from enum import Enum
from typing import List, Optional

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta, VideoMeta
from pydantic import Field

from .pruna_helper import run_prediction, get_generation_url, download_video, file_url, probe_video


class SizeEnum(str, Enum):
    landscape_480 = "832*480"
    portrait_480 = "480*832"
    landscape_720 = "1280*720"
    portrait_720 = "720*1280"


class SpeedModeEnum(str, Enum):
    lightly_juiced = "Lightly Juiced \U0001F34A (more consistent)"
    juiced = "Juiced \U0001F525 (more speed)"
    extra_juiced = "Extra Juiced \U0001F680 (even more speed)"


class SolverEnum(str, Enum):
    unipc = "unipc"
    dpmpp = "dpm++"


class AppInput(BaseAppInput):
    prompt: str = Field(description="Description of the video. Be specific about characters, action, setting and camera.")
    src_video: Optional[File] = Field(default=None, description="Source video to edit or transform.")
    src_mask: Optional[File] = Field(default=None, description="Mask video or image marking the area of the source to edit.")
    src_ref_images: List[File] = Field(
        default_factory=list,
        max_length=3,
        description="Reference images for consistent characters (1-3 work best).",
    )
    size: SizeEnum = Field(default=SizeEnum.landscape_480, description="Output resolution (width*height).")
    frame_num: int = Field(default=81, ge=1, le=81, description="Number of frames to generate (up to 81).")
    speed_mode: SpeedModeEnum = Field(default=SpeedModeEnum.lightly_juiced, description="Speed versus consistency.")
    sample_steps: int = Field(default=50, ge=1, le=100, description="Sampling steps. Higher is better and slower.")
    sample_solver: SolverEnum = Field(default=SolverEnum.unipc, description="Sampling algorithm.")
    sample_guide_scale: float = Field(default=5.0, ge=0, le=20, description="How closely to follow the prompt.")
    sample_shift: int = Field(default=16, ge=0, le=50, description="Sample shift.")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducible generation.")


class AppOutput(BaseAppOutput):
    video: File = Field(description="Generated video.")
    seed: Optional[int] = Field(default=None, description="Seed used for generation.")


class App(BaseApp):
    async def setup(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.model = "vace"

    async def run(self, input_data: AppInput) -> AppOutput:
        request_data = {
            "prompt": input_data.prompt,
            "size": input_data.size.value,
            "frame_num": input_data.frame_num,
            "speed_mode": input_data.speed_mode.value,
            "sample_steps": input_data.sample_steps,
            "sample_solver": input_data.sample_solver.value,
            "sample_guide_scale": input_data.sample_guide_scale,
            "sample_shift": input_data.sample_shift,
            "seed": input_data.seed if input_data.seed is not None else -1,
        }
        if input_data.src_video:
            request_data["src_video"] = file_url(input_data.src_video, logger=self.logger)
        if input_data.src_mask:
            request_data["src_mask"] = file_url(input_data.src_mask, logger=self.logger)
        if input_data.src_ref_images:
            request_data["src_ref_images"] = [file_url(f, logger=self.logger) for f in input_data.src_ref_images]

        self.logger.info("vace: %s, %d frames, video=%s, mask=%s, refs=%d", input_data.size.value, input_data.frame_num,
                         bool(input_data.src_video), bool(input_data.src_mask), len(input_data.src_ref_images))

        result = await run_prediction(model=self.model, input_data=request_data, logger=self.logger)
        self.logger.info("Prediction metadata: %s", result.get("metadata"))
        video_path = download_video(get_generation_url(result), logger=self.logger)

        info = probe_video(video_path, logger=self.logger)
        width, height = (int(x) for x in input_data.size.value.split("*"))
        resolution = "720p" if max(width, height) >= 1280 else "480p"

        inputs = []
        if input_data.src_ref_images:
            inputs.append(ImageMeta(count=len(input_data.src_ref_images)))
        if input_data.src_video:
            inputs.append(VideoMeta())

        # Flat price per video
        output_meta = OutputMeta(
            inputs=inputs,
            outputs=[
                VideoMeta(
                    width=info["width"] or width,
                    height=info["height"] or height,
                    resolution=resolution,
                    seconds=round(info["seconds"], 3),
                    fps=int(round(info["fps"])),
                    extra={"count": 1},
                )
            ],
        )
        return AppOutput(video=File(path=video_path), seed=result.get("seed"), output_meta=output_meta)

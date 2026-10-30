"""
P-Video-2-Pro - Text-to-video and first/last-frame image-to-video by Pruna (MiniMax H3)
"""

import logging
import math
from enum import Enum
from typing import Optional

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta, VideoMeta
from pydantic import Field

from .pruna_helper import run_prediction, get_generation_url, download_video, file_url, probe_video


class ResolutionEnum(str, Enum):
    r480 = "480p"
    r768 = "768p"
    r1080 = "1080p"
    r2k = "2k"
    r4k = "4k"


class ModeEnum(str, Enum):
    cost = "cost"
    speed = "speed"
    quality = "quality"


class PromptUpsamplerEnum(str, Enum):
    off = "off"
    turbo = "turbo"
    max = "max"


class AspectRatioEnum(str, Enum):
    landscape = "16:9"
    portrait = "9:16"
    photo_landscape = "4:3"
    photo_portrait = "3:4"
    classic_landscape = "3:2"
    classic_portrait = "2:3"
    square = "1:1"


class AppInput(BaseAppInput):
    prompt: str = Field(description="Text description for video generation.")
    image: Optional[File] = Field(default=None, description="First-frame image for image-to-video.")
    last_frame_image: Optional[File] = Field(default=None, description="Last-frame image.")
    duration: int = Field(default=5, ge=5, le=15, description="Video duration in seconds (5-15).")
    resolution: ResolutionEnum = Field(
        default=ResolutionEnum.r768,
        description="480p or 768p native. 1080p, 2k and 4k generate at 768p and upscale.",
    )
    mode: ModeEnum = Field(
        default=ModeEnum.speed,
        description="Generation recipe. cost is cheapest and slowest, speed is the default, quality costs most.",
    )
    prompt_upsampler: PromptUpsamplerEnum = Field(
        default=PromptUpsamplerEnum.turbo,
        description="Expand the prompt before generation: off, turbo or max. Does not change the price.",
    )
    aspect_ratio: AspectRatioEnum = Field(
        default=AspectRatioEnum.landscape,
        description="Aspect ratio. Frame images set the canvas when provided.",
    )
    seed: Optional[int] = Field(default=None, description="Random seed for reproducible generation.")
    enable_safety_filter: bool = Field(
        default=False,
        description="Check inputs against the content policy. Always-on categories are blocked either way.",
    )


class AppOutput(BaseAppOutput):
    video: File = Field(description="Generated video with audio.")
    seed: Optional[int] = Field(default=None, description="Seed used for generation.")


class App(BaseApp):
    async def setup(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.model = "p-video-2-pro"

    async def run(self, input_data: AppInput) -> AppOutput:
        request_data = {
            "prompt": input_data.prompt,
            "duration": input_data.duration,
            "resolution": input_data.resolution.value,
            "mode": input_data.mode.value,
            "prompt_upsampler": input_data.prompt_upsampler.value,
            "aspect_ratio": input_data.aspect_ratio.value,
            "enable_safety_filter": input_data.enable_safety_filter,
        }
        if input_data.image:
            request_data["image"] = file_url(input_data.image, logger=self.logger)
        if input_data.last_frame_image:
            request_data["last_frame_image"] = file_url(input_data.last_frame_image, logger=self.logger)
        if input_data.seed is not None:
            request_data["seed"] = input_data.seed

        self.logger.info(
            "p-video-2-pro: %s, mode=%s, %ss, image=%s, last_frame=%s",
            input_data.resolution.value, input_data.mode.value, input_data.duration,
            bool(input_data.image), bool(input_data.last_frame_image),
        )

        result = await run_prediction(model=self.model, input_data=request_data, logger=self.logger)
        self.logger.info("Prediction metadata: %s", result.get("metadata"))
        video_path = download_video(get_generation_url(result), logger=self.logger)

        # Pruna bills whole seconds of the finished video: int(num_frames / 24)
        info = probe_video(video_path, logger=self.logger)
        seconds = float(math.floor(info["seconds"] + 1e-6)) if info["seconds"] > 0 else float(input_data.duration)

        inputs = []
        if input_data.image:
            inputs.append(ImageMeta(count=1))
        if input_data.last_frame_image:
            inputs.append(ImageMeta(count=1))

        output_meta = OutputMeta(
            inputs=inputs,
            outputs=[
                VideoMeta(
                    width=info["width"],
                    height=info["height"],
                    resolution=input_data.resolution.value,
                    seconds=seconds,
                    fps=24,
                    extra={"resolution": input_data.resolution.value, "mode": input_data.mode.value},
                )
            ],
        )
        self.logger.info("Video: %sx%s, %.2fs probed, billed %ss", info["width"], info["height"], info["seconds"], seconds)
        return AppOutput(video=File(path=video_path), seed=result.get("seed"), output_meta=output_meta)

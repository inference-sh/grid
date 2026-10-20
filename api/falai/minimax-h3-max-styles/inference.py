"""
MiniMax H3 Max Styles via fal.ai

fal's stylized H3 Max pipelines: VHS, hand-drawn, 16-bit pixel, retro 70s toon and low-poly.
768P output with audio, from a prompt or a first-frame image.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta, ImageMeta
from pydantic import Field
from typing import Optional, Literal
from enum import Enum
import logging

from .fal_helper import setup_fal_client, run_fal_model, download_video
from .h3max_common import dims_for, probe_video, FPS

logging.getLogger("httpx").setLevel(logging.WARNING)

RESOLUTION = "768P"


class StyleEnum(str, Enum):
    vhs = "vhs"
    hand_drawn = "hand-drawn"
    pixel_16bit = "16bit-pixel"
    retro_toon_70s = "retro-toon-70s"
    low_poly = "low-poly"


ENDPOINTS = {s.value: f"minimax/h3-max/styles/{s.value}" for s in StyleEnum}


class AspectRatioEnum(str, Enum):
    r21_9 = "21:9"
    r16_9 = "16:9"
    r4_3 = "4:3"
    r1_1 = "1:1"
    r3_4 = "3:4"
    r9_16 = "9:16"


class AppInput(BaseAppInput):
    """Mode is determined by the inputs: prompt only is text-to-video, prompt + image is image-to-video."""

    prompt: str = Field(
        description="Describe the scene, action, camera and sound. The chosen style is applied automatically.",
        examples=["A skateboarder cruises down a palm-lined boulevard at sunset, upbeat synth music."],
    )
    style: StyleEnum = Field(default=StyleEnum.vhs, description="Visual style pipeline.")
    image: Optional[File] = Field(
        default=None,
        description="Optional first-frame image. Artwork in the chosen style gives the most consistent look; its aspect ratio sets the canvas.",
    )
    duration: int = Field(default=5, ge=5, le=15, description="Video duration in seconds (5-15).")
    aspect_ratio: AspectRatioEnum = Field(
        default=AspectRatioEnum.r16_9,
        description="Aspect ratio for text-to-video. Ignored when an image is given.",
    )
    damage_level: Literal["light", "medium", "heavy"] = Field(
        default="medium",
        description="VHS style only: tape wear from subtle analog noise (light) to strong tracking errors (heavy).",
    )
    seed: Optional[int] = Field(default=None, description="Random seed for reproducible generation.")


class AppOutput(BaseAppOutput):
    video: File = Field(description="The generated 768P stylized video with audio.")
    seed: Optional[int] = Field(default=None, description="Seed used for this generation.")


class App(BaseApp):

    async def setup(self, metadata):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        self.logger.info("MiniMax H3 Max Styles initialized")

    def _build_request(self, input_data: AppInput) -> dict:
        request = {
            "prompt": input_data.prompt,
            "duration": input_data.duration,
            "aspect_ratio": input_data.aspect_ratio.value,
        }
        if input_data.image:
            request["image_url"] = input_data.image.uri
        if input_data.seed is not None:
            request["seed"] = input_data.seed
        if input_data.style == StyleEnum.vhs:
            request["damage_level"] = input_data.damage_level
        return request

    async def run(self, input_data: AppInput) -> AppOutput:
        setup_fal_client()
        mode = "image-to-video" if input_data.image else "text-to-video"
        endpoint = ENDPOINTS[input_data.style.value]
        self.logger.info(f"Style: {input_data.style.value}, mode: {mode}, duration: {input_data.duration}s, endpoint: {endpoint}")
        result = run_fal_model(endpoint, self._build_request(input_data), self.logger)
        video_path = download_video(result["video"]["url"], self.logger)

        info = probe_video(video_path, self.logger)
        width, height = (info.get("width"), info.get("height")) if info.get("width") else dims_for(RESOLUTION, input_data.aspect_ratio.value)
        output_meta = OutputMeta(
            inputs=[ImageMeta()] if input_data.image else [],
            outputs=[VideoMeta(
                width=int(width), height=int(height), resolution=RESOLUTION,
                seconds=float(input_data.duration), fps=int(info.get("fps") or FPS),
                extra={"style": input_data.style.value, "mode": mode},
            )],
        )
        self.logger.info(f"Generated {width}x{height} {input_data.style.value} video, {input_data.duration}s")
        return AppOutput(video=File(path=video_path), seed=result.get("seed"), output_meta=output_meta)

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field
from typing import Optional
from PIL import Image
import logging

from . import bria_helper

logger = logging.getLogger(__name__)


class AppInput(BaseAppInput):
    image: File = Field(description="Image to expand (JPEG, PNG, WEBP)")
    aspect_ratio: Optional[str] = Field(
        default=None,
        description="Target aspect ratio: predefined ('1:1', '2:3', '3:2', '4:3', '3:4', '16:9', '9:16') or custom float 0.5-3.0. Centers image and expands edges.",
    )
    canvas_size: Optional[list[int]] = Field(
        default=None, description="Output dimensions [width, height] in pixels (max 5000x5000)"
    )
    original_image_size: Optional[list[int]] = Field(
        default=None, description="Desired input size [width, height]. Required when not using aspect_ratio."
    )
    original_image_location: Optional[list[int]] = Field(
        default=None, description="Top-left corner position [x, y] in pixels. Required when not using aspect_ratio."
    )
    prompt: Optional[str] = Field(default=None, description="English text guidance for the expanded area")
    negative_prompt: Optional[str] = Field(default=None, description="English text specifying unwanted elements")
    seed: Optional[int] = Field(default=None, description="Seed for reproducible results")
    preserve_alpha: Optional[bool] = Field(default=None, description="Retain alpha channel transparency")


class AppOutput(BaseAppOutput):
    image: File = Field(description="Expanded image")
    seed: Optional[int] = Field(default=None, description="Seed used for generation")
    prompt: Optional[str] = Field(default=None, description="Prompt used (may be auto-generated)")


class App(BaseApp):
    async def setup(self, config):
        self.client = bria_helper.get_client()
        logger.info("Bria Expand ready")

    async def run(self, input_data: AppInput) -> AppOutput:
        payload = {"image": input_data.image.uri}

        for key in ("aspect_ratio", "canvas_size", "original_image_size", "original_image_location", "prompt", "negative_prompt", "seed", "preserve_alpha"):
            val = getattr(input_data, key)
            if val is not None:
                payload[key] = val

        logger.info(f"Requesting image expansion (aspect_ratio={input_data.aspect_ratio})")
        result = await bria_helper.call_endpoint(self.client, "expand", payload)

        image_url = result["result"]["image_url"]
        seed = result["result"].get("seed")
        used_prompt = result["result"].get("prompt")
        path = await bria_helper.download_image(self.client, image_url)
        logger.info(f"Downloaded expanded image to {path}")

        with Image.open(path) as img:
            width, height = img.size

        return AppOutput(
            image=File(path=path),
            seed=int(seed) if seed else None,
            prompt=used_prompt,
            output_meta=OutputMeta(outputs=[ImageMeta(width=width, height=height, count=1)]),
        )

    async def unload(self):
        await self.client.aclose()

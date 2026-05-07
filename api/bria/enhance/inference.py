from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field
from typing import Optional
from PIL import Image
import logging

from . import bria_helper

logger = logging.getLogger(__name__)


class AppInput(BaseAppInput):
    image: File = Field(description="Image to enhance (JPEG, PNG, WEBP)")
    seed: Optional[int] = Field(default=None, description="Seed for reproducible results")
    preserve_alpha: Optional[bool] = Field(default=None, description="Retain alpha channel transparency")


class AppOutput(BaseAppOutput):
    image: File = Field(description="Enhanced image with doubled resolution and added detail")
    seed: Optional[int] = Field(default=None, description="Seed used for generation")


class App(BaseApp):
    async def setup(self, config):
        self.client = bria_helper.get_client()
        logger.info("Bria Enhance ready")

    async def run(self, input_data: AppInput) -> AppOutput:
        payload = {"image": input_data.image.uri}
        if input_data.seed is not None:
            payload["seed"] = input_data.seed
        if input_data.preserve_alpha is not None:
            payload["preserve_alpha"] = input_data.preserve_alpha

        logger.info("Requesting image enhancement")
        result = await bria_helper.call_endpoint(self.client, "enhance", payload)

        image_url = result["result"]["image_url"]
        seed = result["result"].get("seed")
        path = await bria_helper.download_image(self.client, image_url)
        logger.info(f"Downloaded enhanced image to {path}")

        with Image.open(path) as img:
            width, height = img.size

        return AppOutput(
            image=File(path=path),
            seed=int(seed) if seed else None,
            output_meta=OutputMeta(outputs=[ImageMeta(width=width, height=height, count=1)]),
        )

    async def unload(self):
        await self.client.aclose()

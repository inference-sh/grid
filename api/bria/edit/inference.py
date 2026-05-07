from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field
from typing import Optional
from PIL import Image
import logging

from . import bria_helper

logger = logging.getLogger(__name__)

V2_BASE = "https://engine.prod.bria-api.com/v2"


class AppInput(BaseAppInput):
    image: File = Field(description="Source image (JPEG, PNG, WEBP)")
    instruction: str = Field(description="Text instruction describing the desired edit")
    seed: Optional[int] = Field(default=None, description="Seed for reproducible results")


class AppOutput(BaseAppOutput):
    image: File = Field(description="Edited image")


class App(BaseApp):
    async def setup(self, config):
        self.client = bria_helper.get_client()
        logger.info("Bria Edit ready")

    async def run(self, input_data: AppInput) -> AppOutput:
        payload = {
            "images": [input_data.image.uri],
            "instruction": input_data.instruction,
        }
        if input_data.seed is not None:
            payload["seed"] = input_data.seed

        logger.info(f"Requesting image edit: {input_data.instruction[:80]}")
        result = await bria_helper.call_endpoint(self.client, "image/edit", payload, base_url=V2_BASE)

        image_url = result["result"]["image_url"]
        path = await bria_helper.download_image(self.client, image_url)
        logger.info(f"Downloaded edited image to {path}")

        with Image.open(path) as img:
            width, height = img.size

        return AppOutput(
            image=File(path=path),
            output_meta=OutputMeta(outputs=[ImageMeta(width=width, height=height, count=1)]),
        )

    async def unload(self):
        await self.client.aclose()

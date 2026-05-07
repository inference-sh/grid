from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field
from typing import Optional, Literal
from PIL import Image
import logging

from . import bria_helper

logger = logging.getLogger(__name__)


class AppInput(BaseAppInput):
    image: File = Field(description="Image to upscale (JPEG, PNG, WEBP)")
    desired_increase: Literal[2, 4] = Field(default=2, description="Resolution multiplier: 2x or 4x (max output 8192x8192)")
    preserve_alpha: Optional[bool] = Field(default=None, description="Retain alpha channel transparency")


class AppOutput(BaseAppOutput):
    image: File = Field(description="Upscaled image")


class App(BaseApp):
    async def setup(self, config):
        self.client = bria_helper.get_client()
        logger.info("Bria Increase Resolution ready")

    async def run(self, input_data: AppInput) -> AppOutput:
        payload = {
            "image": input_data.image.uri,
            "desired_increase": input_data.desired_increase,
        }
        if input_data.preserve_alpha is not None:
            payload["preserve_alpha"] = input_data.preserve_alpha

        logger.info(f"Requesting {input_data.desired_increase}x upscale")
        result = await bria_helper.call_endpoint(self.client, "increase_resolution", payload)

        image_url = result["result"]["image_url"]
        path = await bria_helper.download_image(self.client, image_url)
        logger.info(f"Downloaded result to {path}")

        with Image.open(path) as img:
            width, height = img.size

        return AppOutput(
            image=File(path=path),
            output_meta=OutputMeta(outputs=[ImageMeta(width=width, height=height, count=1)]),
        )

    async def unload(self):
        await self.client.aclose()

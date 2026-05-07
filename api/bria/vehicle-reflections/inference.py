from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field
from typing import Optional
from PIL import Image
import logging

from . import bria_helper

logger = logging.getLogger(__name__)

PRODUCT_BASE = "https://engine.prod.bria-api.com/v1/product"


class AppInput(BaseAppInput):
    image: File = Field(description="Vehicle image to add reflections to (JPEG, PNG, WEBP, max 12MB)")
    content_moderation: Optional[bool] = Field(default=None, description="Apply content moderation to input and output images")


class AppOutput(BaseAppOutput):
    image: File = Field(description="Vehicle image with realistic reflections added")


class App(BaseApp):
    async def setup(self, config):
        self.client = bria_helper.get_client()
        logger.info("Bria Vehicle Reflections ready")

    async def run(self, input_data: AppInput) -> AppOutput:
        payload = {"image_url": input_data.image.uri}

        if input_data.content_moderation is not None:
            payload["content_moderation"] = input_data.content_moderation

        logger.info("Requesting vehicle reflections")
        result = await bria_helper.call_endpoint(self.client, "vehicle/generate_reflections", payload, base_url=PRODUCT_BASE)

        image_url = result["result"]["image_url"] if isinstance(result["result"], dict) else result["result"]
        path = await bria_helper.download_image(self.client, image_url)
        logger.info(f"Downloaded reflections image to {path}")

        with Image.open(path) as img:
            width, height = img.size

        return AppOutput(
            image=File(path=path),
            output_meta=OutputMeta(outputs=[ImageMeta(width=width, height=height, count=1)]),
        )

    async def unload(self):
        await self.client.aclose()

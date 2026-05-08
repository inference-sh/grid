from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field
from typing import Optional
from PIL import Image
import logging

from . import bria_helper

logger = logging.getLogger(__name__)

PRODUCT_BASE = "https://engine.prod.bria-api.com/v1/product"


class AppInput(BaseAppInput):
    image: File = Field(description="Vehicle composite image to harmonize lighting (JPEG, PNG, WEBP, max 12MB)")
    preset: str = Field(description="Harmonization preset: 'warm day', 'cold day', 'warm night', 'cold night'")
    content_moderation: Optional[bool] = Field(default=None, description="Apply content moderation to input and output images")


class AppOutput(BaseAppOutput):
    image: File = Field(description="Vehicle image with harmonized lighting and colors")


class App(BaseApp):
    async def setup(self, config):
        self.client = bria_helper.get_client()
        logger.info("Bria Vehicle Harmonize ready")

    async def run(self, input_data: AppInput) -> AppOutput:
        payload = {
            "image_url": input_data.image.uri,
            "preset": input_data.preset,
        }
        if input_data.content_moderation is not None:
            payload["content_moderation"] = input_data.content_moderation

        logger.info("Requesting vehicle harmonization")
        result = await bria_helper.call_endpoint(self.client, "vehicle/harmonize", payload, base_url=PRODUCT_BASE)

        r = result.get("result_url") or result.get("result")
        image_url = r[0] if isinstance(r, list) else (r.get("image_url") if isinstance(r, dict) else r)
        path = await bria_helper.download_image(self.client, image_url)
        logger.info(f"Downloaded harmonized image to {path}")

        with Image.open(path) as img:
            width, height = img.size

        return AppOutput(
            image=File(path=path),
            output_meta=OutputMeta(outputs=[ImageMeta(width=width, height=height, count=1)]),
        )

    async def unload(self):
        await self.client.aclose()

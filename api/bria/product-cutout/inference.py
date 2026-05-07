from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field
from typing import Optional
from PIL import Image
import logging

from . import bria_helper

logger = logging.getLogger(__name__)

PRODUCT_BASE = "https://engine.prod.bria-api.com/v1/product"


class AppInput(BaseAppInput):
    image: File = Field(description="Product image to cut out (JPEG, PNG, WEBP, max 12MB)")
    content_moderation: Optional[bool] = Field(default=None, description="Apply content moderation to input and output images")
    force_rmbg: Optional[bool] = Field(default=None, description="Force background removal even if image has alpha channel")
    preserve_alpha: Optional[bool] = Field(default=None, description="Retain input alpha channel transparency in output")


class AppOutput(BaseAppOutput):
    image: File = Field(description="Product cutout image with transparent background")


class App(BaseApp):
    async def setup(self, config):
        self.client = bria_helper.get_client()
        logger.info("Bria Product Cutout ready")

    async def run(self, input_data: AppInput) -> AppOutput:
        payload = {"image": input_data.image.uri}

        for key in ("content_moderation", "force_rmbg", "preserve_alpha"):
            val = getattr(input_data, key)
            if val is not None:
                payload[key] = val

        logger.info("Requesting product cutout")
        result = await bria_helper.call_endpoint(self.client, "cutout", payload, base_url=PRODUCT_BASE)

        image_url = result["result"]["image_url"]
        path = await bria_helper.download_image(self.client, image_url)
        logger.info(f"Downloaded cutout to {path}")

        with Image.open(path) as img:
            width, height = img.size

        return AppOutput(
            image=File(path=path),
            output_meta=OutputMeta(outputs=[ImageMeta(width=width, height=height, count=1)]),
        )

    async def unload(self):
        await self.client.aclose()

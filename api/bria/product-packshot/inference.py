from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field
from typing import Optional
from PIL import Image
import logging

from . import bria_helper

logger = logging.getLogger(__name__)

PRODUCT_BASE = "https://engine.prod.bria-api.com/v1/product"


class AppInput(BaseAppInput):
    image: File = Field(description="Product image or cutout (JPEG, PNG, WEBP, max 12MB)")
    background_color: Optional[str] = Field(default=None, description="Background hex color (e.g. '#FFFFFF') or 'transparent'")
    content_moderation: Optional[bool] = Field(default=None, description="Apply content moderation to input and output images")
    force_rmbg: Optional[bool] = Field(default=None, description="Force background removal even if image has alpha channel")


class AppOutput(BaseAppOutput):
    image: File = Field(description="Professional 2000x2000 product packshot")


class App(BaseApp):
    async def setup(self, config):
        self.client = bria_helper.get_client()
        logger.info("Bria Product Packshot ready")

    async def run(self, input_data: AppInput) -> AppOutput:
        payload = {"image_url": input_data.image.uri}

        for key in ("background_color", "content_moderation", "force_rmbg"):
            val = getattr(input_data, key)
            if val is not None:
                payload[key] = val

        logger.info("Requesting product packshot")
        result = await bria_helper.call_endpoint(self.client, "packshot", payload, base_url=PRODUCT_BASE)

        image_url = result["result"]["image_url"]
        path = await bria_helper.download_image(self.client, image_url)
        logger.info(f"Downloaded packshot to {path}")

        with Image.open(path) as img:
            width, height = img.size

        return AppOutput(
            image=File(path=path),
            output_meta=OutputMeta(outputs=[ImageMeta(width=width, height=height, count=1)]),
        )

    async def unload(self):
        await self.client.aclose()

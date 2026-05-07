from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field
from typing import Optional
from PIL import Image
import logging

from . import bria_helper

logger = logging.getLogger(__name__)

EDIT_BASE = "https://engine.prod.bria-api.com/v1/image/edit"


class AppInput(BaseAppInput):
    image: File = Field(description="Product image or cutout (JPEG, PNG, WEBP, max 12MB)")
    scene: File = Field(description="Scene image to embed the product into")
    placement: Optional[dict] = Field(default=None, description="Product placement: {'x': int, 'y': int, 'width': int, 'height': int}")
    num_results: Optional[int] = Field(default=None, description="Number of result images to generate (1-4)")
    seed: Optional[int] = Field(default=None, description="Seed for reproducible results")
    content_moderation: Optional[bool] = Field(default=None, description="Apply content moderation to input and output images")
    force_rmbg: Optional[bool] = Field(default=None, description="Force background removal on product image")


class AppOutput(BaseAppOutput):
    image: File = Field(description="Scene with product embedded")


class App(BaseApp):
    async def setup(self, config):
        self.client = bria_helper.get_client()
        logger.info("Bria Product Embed ready")

    async def run(self, input_data: AppInput) -> AppOutput:
        payload = {
            "image": input_data.image.uri,
            "scene": input_data.scene.uri,
        }

        for key in ("placement", "num_results", "seed", "content_moderation", "force_rmbg"):
            val = getattr(input_data, key)
            if val is not None:
                payload[key] = val

        logger.info("Requesting product embed")
        result = await bria_helper.call_endpoint(self.client, "product/integrate", payload, base_url=EDIT_BASE)

        image_url = result["result"][0] if isinstance(result["result"], list) else result["result"]["image_url"]
        path = await bria_helper.download_image(self.client, image_url)
        logger.info(f"Downloaded embedded image to {path}")

        with Image.open(path) as img:
            width, height = img.size

        return AppOutput(
            image=File(path=path),
            output_meta=OutputMeta(outputs=[ImageMeta(width=width, height=height, count=1)]),
        )

    async def unload(self):
        await self.client.aclose()

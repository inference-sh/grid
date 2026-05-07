from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field
from typing import Optional
from PIL import Image
import logging

from . import bria_helper

logger = logging.getLogger(__name__)

PRODUCT_BASE = "https://engine.prod.bria-api.com/v1/product"


class AppInput(BaseAppInput):
    image: File = Field(description="Vehicle image (JPEG, PNG, WEBP, max 12MB)")
    scene_description: str = Field(description="Text description of the desired environment/scene")
    optimize_description: Optional[bool] = Field(default=None, description="Let the API optimize the scene description for better results")
    num_results: Optional[int] = Field(default=None, description="Number of result images to generate (1-4)")
    seed: Optional[int] = Field(default=None, description="Seed for reproducible results")
    content_moderation: Optional[bool] = Field(default=None, description="Apply content moderation to input and output images")


class AppOutput(BaseAppOutput):
    image: File = Field(description="Vehicle placed in the described environment")


class App(BaseApp):
    async def setup(self, config):
        self.client = bria_helper.get_client()
        logger.info("Bria Vehicle Shot (Text) ready")

    async def run(self, input_data: AppInput) -> AppOutput:
        payload = {
            "image_url": input_data.image.uri,
            "scene_description": input_data.scene_description,
        }

        for key in ("optimize_description", "num_results", "seed", "content_moderation"):
            val = getattr(input_data, key)
            if val is not None:
                payload[key] = val

        logger.info("Requesting vehicle shot by text")
        result = await bria_helper.call_endpoint(self.client, "vehicle/shot_by_text", payload, base_url=PRODUCT_BASE)

        image_url = bria_helper.get_result_url(result)
        path = await bria_helper.download_image(self.client, image_url)
        logger.info(f"Downloaded vehicle shot to {path}")

        with Image.open(path) as img:
            width, height = img.size

        return AppOutput(
            image=File(path=path),
            output_meta=OutputMeta(outputs=[ImageMeta(width=width, height=height, count=1)]),
        )

    async def unload(self):
        await self.client.aclose()

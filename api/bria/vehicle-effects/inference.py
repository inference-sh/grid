from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field
from typing import Optional, Literal
from PIL import Image
import logging

from . import bria_helper

logger = logging.getLogger(__name__)

PRODUCT_BASE = "https://engine.prod.bria-api.com/v1/product"


class AppInput(BaseAppInput):
    image: File = Field(description="Vehicle image to apply effects to (JPEG, PNG, WEBP, max 12MB)")
    effect: Literal["dust", "snow", "fog", "rain", "smoke", "puddle", "wet_road"] = Field(
        description="Effect to apply: dust, snow, fog, rain, smoke, puddle, or wet_road"
    )
    intensity: Optional[float] = Field(default=None, description="Effect intensity from 0.0 to 1.0")
    content_moderation: Optional[bool] = Field(default=None, description="Apply content moderation to input and output images")


class AppOutput(BaseAppOutput):
    image: File = Field(description="Vehicle image with effect applied")


class App(BaseApp):
    async def setup(self, config):
        self.client = bria_helper.get_client()
        logger.info("Bria Vehicle Effects ready")

    async def run(self, input_data: AppInput) -> AppOutput:
        payload = {
            "image_url": input_data.image.uri,
            "effect": input_data.effect,
        }

        for key in ("intensity", "content_moderation"):
            val = getattr(input_data, key)
            if val is not None:
                payload[key] = val

        logger.info(f"Requesting vehicle effect: {input_data.effect}")
        result = await bria_helper.call_endpoint(self.client, "vehicle/apply_effect", payload, base_url=PRODUCT_BASE)

        image_url = bria_helper.get_result_url(result) if isinstance(result["result"], dict) else result["result"]
        path = await bria_helper.download_image(self.client, image_url)
        logger.info(f"Downloaded effects image to {path}")

        with Image.open(path) as img:
            width, height = img.size

        return AppOutput(
            image=File(path=path),
            output_meta=OutputMeta(outputs=[ImageMeta(width=width, height=height, count=1)]),
        )

    async def unload(self):
        await self.client.aclose()

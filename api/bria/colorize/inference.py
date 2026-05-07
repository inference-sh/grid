from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field
from typing import Optional, Literal
from PIL import Image
import logging

from . import bria_helper

logger = logging.getLogger(__name__)

V2_BASE = "https://engine.prod.bria-api.com/v2"


class AppInput(BaseAppInput):
    image: File = Field(description="Black and white or grayscale image (JPEG, PNG, WEBP)")
    color: Literal["color_contemporary", "decolorize", "sepia_vintage"] = Field(
        description="Color style: 'color_contemporary' (natural), 'decolorize' (remove color), 'sepia_vintage' (aged look)"
    )


class AppOutput(BaseAppOutput):
    image: File = Field(description="Colorized image")


class App(BaseApp):
    async def setup(self, config):
        self.client = bria_helper.get_client()
        logger.info("Bria Colorize ready")

    async def run(self, input_data: AppInput) -> AppOutput:
        payload = {
            "image": input_data.image.uri,
            "color": input_data.color,
        }

        logger.info("Requesting colorization")
        result = await bria_helper.call_endpoint(self.client, "colorize", payload)

        image_url = result["result"]["image_url"]
        path = await bria_helper.download_image(self.client, image_url)
        logger.info(f"Downloaded colorized image to {path}")

        with Image.open(path) as img:
            width, height = img.size

        return AppOutput(
            image=File(path=path),
            output_meta=OutputMeta(outputs=[ImageMeta(width=width, height=height, count=1)]),
        )

    async def unload(self):
        await self.client.aclose()

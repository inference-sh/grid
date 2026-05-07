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
    style: str = Field(description="Style preset ID: render_3d, cubism, oil_painting, anime, cartoon, coloring_book, retro_ad, pop_art_halftone, vector_art, story_board, art_nouveau, cross_etching, wood_cut")
    seed: Optional[int] = Field(default=None, description="Seed for reproducible results (0-2147483647)")


class AppOutput(BaseAppOutput):
    image: File = Field(description="Restyled image")


class App(BaseApp):
    async def setup(self, config):
        self.client = bria_helper.get_client()
        logger.info("Bria Restyle ready")

    async def run(self, input_data: AppInput) -> AppOutput:
        payload = {
            "image": input_data.image.uri,
            "style": input_data.style,
        }
        if input_data.seed is not None:
            payload["seed"] = input_data.seed

        logger.info("Requesting restyle")
        result = await bria_helper.call_endpoint(self.client, "restyle", payload)

        image_url = result["result"]["image_url"]
        path = await bria_helper.download_image(self.client, image_url)
        logger.info(f"Downloaded restyled image to {path}")

        with Image.open(path) as img:
            width, height = img.size

        return AppOutput(
            image=File(path=path),
            output_meta=OutputMeta(outputs=[ImageMeta(width=width, height=height, count=1)]),
        )

    async def unload(self):
        await self.client.aclose()

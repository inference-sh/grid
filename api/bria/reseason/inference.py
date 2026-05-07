from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field
from typing import Optional, Literal
from PIL import Image
import logging

from . import bria_helper

logger = logging.getLogger(__name__)


class AppInput(BaseAppInput):
    image: File = Field(description="Source image (JPEG, PNG, WEBP)")
    season: Literal["spring", "summer", "autumn", "winter"] = Field(description="Target season to apply")
    seed: Optional[int] = Field(default=None, description="Seed for reproducible results (0-2147483647)")


class AppOutput(BaseAppOutput):
    image: File = Field(description="Image with changed season")


class App(BaseApp):
    async def setup(self, config):
        self.client = bria_helper.get_client()
        logger.info("Bria Reseason ready")

    async def run(self, input_data: AppInput) -> AppOutput:
        payload = {
            "image": input_data.image.uri,
            "season": input_data.season,
        }
        if input_data.seed is not None:
            payload["seed"] = input_data.seed

        logger.info(f"Requesting reseason to {input_data.season}")
        result = await bria_helper.call_endpoint(self.client, "reseason", payload)

        image_url = result["result"]["image_url"]
        path = await bria_helper.download_image(self.client, image_url)
        logger.info(f"Downloaded reseasoned image to {path}")

        with Image.open(path) as img:
            width, height = img.size

        return AppOutput(
            image=File(path=path),
            output_meta=OutputMeta(outputs=[ImageMeta(width=width, height=height, count=1)]),
        )

    async def unload(self):
        await self.client.aclose()

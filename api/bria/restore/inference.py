from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field
from PIL import Image
import logging

from . import bria_helper

logger = logging.getLogger(__name__)


class AppInput(BaseAppInput):
    image: File = Field(description="Old or degraded image to restore (JPEG, PNG, WEBP)")


class AppOutput(BaseAppOutput):
    image: File = Field(description="Restored image")


class App(BaseApp):
    async def setup(self, config):
        self.client = bria_helper.get_client()
        logger.info("Bria Restore ready")

    async def run(self, input_data: AppInput) -> AppOutput:
        payload = {"image": input_data.image.uri}

        logger.info("Requesting image restoration")
        result = await bria_helper.call_endpoint(self.client, "restore", payload)

        image_url = result["result"]["image_url"]
        path = await bria_helper.download_image(self.client, image_url)
        logger.info(f"Downloaded restored image to {path}")

        with Image.open(path) as img:
            width, height = img.size

        return AppOutput(
            image=File(path=path),
            output_meta=OutputMeta(outputs=[ImageMeta(width=width, height=height, count=1)]),
        )

    async def unload(self):
        await self.client.aclose()

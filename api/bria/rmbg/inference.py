from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field
from typing import Optional
from PIL import Image
import logging

from . import bria_helper

logger = logging.getLogger(__name__)


class AppInput(BaseAppInput):
    image: File = Field(description="Image to remove background from (JPEG, PNG, WEBP)")
    preserve_alpha: Optional[bool] = Field(default=None, description="Retain partially transparent areas from the input")


class AppOutput(BaseAppOutput):
    image: File = Field(description="Image with background removed")


class App(BaseApp):
    async def setup(self, config):
        self.client = bria_helper.get_client()
        logger.info("Bria RMBG ready")

    async def run(self, input_data: AppInput) -> AppOutput:
        payload = {"image": input_data.image.uri}
        if input_data.preserve_alpha is not None:
            payload["preserve_alpha"] = input_data.preserve_alpha

        logger.info("Requesting background removal")
        result = await bria_helper.call_endpoint(self.client, "remove_background", payload)

        image_url = result["result"]["image_url"]
        path = await bria_helper.download_image(self.client, image_url)
        logger.info(f"Downloaded result to {path}")

        with Image.open(path) as img:
            width, height = img.size

        return AppOutput(
            image=File(path=path),
            output_meta=OutputMeta(outputs=[ImageMeta(width=width, height=height, count=1)]),
        )

    async def unload(self):
        await self.client.aclose()

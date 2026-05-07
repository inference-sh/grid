from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field
from typing import Optional, Literal
from PIL import Image
import logging

from . import bria_helper

logger = logging.getLogger(__name__)


class AppInput(BaseAppInput):
    image: File = Field(description="Source image (JPEG, PNG, WEBP)")
    mask: File = Field(description="Binary mask defining erasure region (white=erase, black=keep). Must match image aspect ratio.")
    mask_type: Optional[Literal["manual", "automatic"]] = Field(
        default=None, description="Mask type: 'manual' (user-drawn) or 'automatic' (algorithm-generated)"
    )
    preserve_alpha: Optional[bool] = Field(default=None, description="Retain alpha channel transparency")


class AppOutput(BaseAppOutput):
    image: File = Field(description="Image with masked region erased and inpainted")


class App(BaseApp):
    async def setup(self, config):
        self.client = bria_helper.get_client()
        logger.info("Bria Eraser ready")

    async def run(self, input_data: AppInput) -> AppOutput:
        payload = {
            "image": input_data.image.uri,
            "mask": input_data.mask.uri,
        }
        if input_data.mask_type is not None:
            payload["mask_type"] = input_data.mask_type
        if input_data.preserve_alpha is not None:
            payload["preserve_alpha"] = input_data.preserve_alpha

        logger.info("Requesting object erasure")
        result = await bria_helper.call_endpoint(self.client, "erase", payload)

        image_url = result["result"]["image_url"]
        path = await bria_helper.download_image(self.client, image_url)
        logger.info(f"Downloaded erased image to {path}")

        with Image.open(path) as img:
            width, height = img.size

        return AppOutput(
            image=File(path=path),
            output_meta=OutputMeta(outputs=[ImageMeta(width=width, height=height, count=1)]),
        )

    async def unload(self):
        await self.client.aclose()

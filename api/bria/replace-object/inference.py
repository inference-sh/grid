from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field
from typing import Optional
from PIL import Image
import logging

from . import bria_helper

logger = logging.getLogger(__name__)


class AppInput(BaseAppInput):
    image: File = Field(description="Source image (JPEG, PNG, WEBP)")
    instruction: str = Field(description="Text description of the replacement object")
    object_to_replace: str = Field(description="Text description of the object to replace")
    negative_prompt: Optional[str] = Field(default=None, description="Text specifying unwanted elements")
    seed: Optional[int] = Field(default=None, description="Seed for reproducible results (0-2147483647)")
    prompt_content_moderation: Optional[bool] = Field(default=None, description="Return 422 if prompt fails moderation. Default: true")


class AppOutput(BaseAppOutput):
    image: File = Field(description="Image with replaced object")


class App(BaseApp):
    async def setup(self, config):
        self.client = bria_helper.get_client()
        logger.info("Bria Replace Object ready")

    async def run(self, input_data: AppInput) -> AppOutput:
        payload = {
            "image": input_data.image.uri,
            "instruction": input_data.instruction,
            "object_to_replace": input_data.object_to_replace,
        }
        if input_data.negative_prompt is not None:
            payload["negative_prompt"] = input_data.negative_prompt
        if input_data.seed is not None:
            payload["seed"] = input_data.seed
        if input_data.prompt_content_moderation is not None:
            payload["prompt_content_moderation"] = input_data.prompt_content_moderation

        logger.info("Requesting object replacement")
        result = await bria_helper.call_endpoint(self.client, "replace_object_by_text", payload)

        image_url = result["result"]["image_url"]
        path = await bria_helper.download_image(self.client, image_url)
        logger.info(f"Downloaded image to {path}")

        with Image.open(path) as img:
            width, height = img.size

        return AppOutput(
            image=File(path=path),
            output_meta=OutputMeta(outputs=[ImageMeta(width=width, height=height, count=1)]),
        )

    async def unload(self):
        await self.client.aclose()

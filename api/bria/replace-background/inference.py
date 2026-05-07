from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field
from typing import Optional, Literal
from PIL import Image
import logging

from . import bria_helper

logger = logging.getLogger(__name__)


class AppInput(BaseAppInput):
    image: File = Field(description="Source image (JPEG, PNG, WEBP)")
    prompt: Optional[str] = Field(default=None, description="Text description of desired background (English, 50-110 words). Either prompt or ref_image required.")
    ref_image: Optional[File] = Field(default=None, description="Reference image for background style. Either prompt or ref_image required.")
    mode: Optional[Literal["base", "high_control", "fast"]] = Field(default=None, description="Generation mode. Default: fast")
    seed: Optional[int] = Field(default=None, description="Seed for reproducible results (0-2147483647)")
    original_quality: Optional[bool] = Field(default=None, description="Preserve original input dimensions. Default: false")
    refine_prompt: Optional[bool] = Field(default=None, description="Auto-adjust prompt for optimal results. Default: true")
    prompt_content_moderation: Optional[bool] = Field(default=None, description="Return 422 if prompt fails moderation. Default: true")


class AppOutput(BaseAppOutput):
    image: File = Field(description="Image with replaced background")


class App(BaseApp):
    async def setup(self, config):
        self.client = bria_helper.get_client()
        logger.info("Bria Replace Background ready")

    async def run(self, input_data: AppInput) -> AppOutput:
        payload = {"image": input_data.image.uri}

        if input_data.prompt is not None:
            payload["prompt"] = input_data.prompt
        if input_data.ref_image is not None:
            payload["ref_image"] = input_data.ref_image.uri
        for key in ("mode", "seed", "original_quality", "refine_prompt", "prompt_content_moderation"):
            val = getattr(input_data, key)
            if val is not None:
                payload[key] = val

        logger.info("Requesting background replacement")
        result = await bria_helper.call_endpoint(self.client, "replace_background", payload)

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

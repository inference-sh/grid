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
    light_type: str = Field(description="Lighting style: 'midday', 'blue hour light', 'low-angle sunlight', 'sunrise light', 'spotlight on subject', 'overcast light', 'soft overcast daylight lighting', 'cloud-filtered lighting', 'fog-diffused lighting', 'side lighting', 'moonlight lighting', 'starlight nighttime', 'soft bokeh lighting', 'harsh studio lighting'")
    light_direction: Optional[str] = Field(default=None, description="Light direction: front, side, bottom, top_down")
    seed: Optional[int] = Field(default=None, description="Seed for reproducible results (0-2147483647)")


class AppOutput(BaseAppOutput):
    image: File = Field(description="Relit image")


class App(BaseApp):
    async def setup(self, config):
        self.client = bria_helper.get_client()
        logger.info("Bria Relight ready")

    async def run(self, input_data: AppInput) -> AppOutput:
        payload = {
            "image": input_data.image.uri,
            "light_type": input_data.light_type,
        }
        if input_data.light_direction is not None:
            payload["light_direction"] = input_data.light_direction
        if input_data.seed is not None:
            payload["seed"] = input_data.seed

        logger.info("Requesting relight")
        result = await bria_helper.call_endpoint(self.client, "relight", payload)

        image_url = result["result"]["image_url"]
        path = await bria_helper.download_image(self.client, image_url)
        logger.info(f"Downloaded relit image to {path}")

        with Image.open(path) as img:
            width, height = img.size

        return AppOutput(
            image=File(path=path),
            output_meta=OutputMeta(outputs=[ImageMeta(width=width, height=height, count=1)]),
        )

    async def unload(self):
        await self.client.aclose()

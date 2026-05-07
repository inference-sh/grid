from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field
from typing import Optional
from PIL import Image
import logging

from . import bria_helper

logger = logging.getLogger(__name__)

GENERATE_BASE = "https://engine.prod.bria-api.com/v2/image"


class AppInput(BaseAppInput):
    prompt: Optional[str] = Field(default=None, description="Text instruction for image generation")
    images: Optional[list[File]] = Field(default=None, description="Reference images for guided generation")
    num_results: int = Field(default=1, description="Number of images to generate")
    sync: bool = Field(default=False, description="Wait for result synchronously")
    aspect_ratio: Optional[str] = Field(default=None, description="Output aspect ratio (e.g. 1:1, 16:9)")
    resolution: Optional[str] = Field(default=None, description="Output resolution: 1MP or 4MP")
    guidance_scale: Optional[int] = Field(default=None, description="How closely to follow prompt (3-5)")
    steps_num: Optional[int] = Field(default=None, description="Diffusion iterations (1-50)")
    seed: Optional[int] = Field(default=None, description="Seed for reproducibility (0-2147483647)")
    output_type: Optional[str] = Field(default=None, description="Output format: png or jpeg")


class AppOutput(BaseAppOutput):
    image: File = Field(description="Generated image")
    structured_prompt: Optional[str] = Field(default=None, description="Structured prompt used for generation")
    seed: Optional[int] = Field(default=None, description="Seed used for generation")


class App(BaseApp):
    async def setup(self, config):
        self.client = bria_helper.get_client()
        logger.info("Bria Generate ready")

    async def run(self, input_data: AppInput) -> AppOutput:
        payload = {}
        if input_data.prompt is not None:
            payload["prompt"] = input_data.prompt
        if input_data.images:
            payload["image_urls"] = [img.uri for img in input_data.images]
        if input_data.num_results != 1:
            payload["num_results"] = input_data.num_results
        if input_data.aspect_ratio is not None:
            payload["aspect_ratio"] = input_data.aspect_ratio
        if input_data.resolution is not None:
            payload["resolution"] = input_data.resolution
        if input_data.guidance_scale is not None:
            payload["guidance_scale"] = input_data.guidance_scale
        if input_data.steps_num is not None:
            payload["steps_num"] = input_data.steps_num
        if input_data.seed is not None:
            payload["seed"] = input_data.seed
        if input_data.output_type is not None:
            payload["output_type"] = input_data.output_type
        if input_data.sync:
            payload["sync"] = True

        logger.info("Requesting image generation")
        result = await bria_helper.call_endpoint(
            self.client, "generate", payload, base_url=GENERATE_BASE
        )

        image_url = result["result"]["image_url"]
        structured_prompt = result["result"].get("structured_prompt")
        seed = result["result"].get("seed")
        path = await bria_helper.download_image(self.client, image_url)
        logger.info(f"Downloaded generated image to {path}")

        with Image.open(path) as img:
            width, height = img.size

        return AppOutput(
            image=File(path=path),
            structured_prompt=structured_prompt,
            seed=int(seed) if seed else None,
            output_meta=OutputMeta(outputs=[ImageMeta(width=width, height=height, count=1)]),
        )

    async def unload(self):
        await self.client.aclose()

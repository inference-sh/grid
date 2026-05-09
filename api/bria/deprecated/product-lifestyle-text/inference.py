from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field
from typing import Optional
from PIL import Image
import logging

from . import bria_helper

logger = logging.getLogger(__name__)

PRODUCT_BASE = "https://engine.prod.bria-api.com/v1/product"


class AppInput(BaseAppInput):
    image: File = Field(description="Product image or cutout — auto-removes background if present, or use product-cutout first for best results (JPEG, PNG, WEBP, max 12MB)")
    scene_description: str = Field(description="Text description of the desired lifestyle scene")
    optimize_description: Optional[bool] = Field(default=None, description="Let the API optimize the scene description for better results")
    placement_type: Optional[str] = Field(default=None, description="How to place the product: 'original' (keep size/position), 'automatic' (auto-fit), or 'manual_placement'")
    manual_padding: Optional[dict] = Field(default=None, description="Manual padding when placement_type='manual_placement': {'left': int, 'right': int, 'top': int, 'bottom': int}")
    shot_size: Optional[list[int]] = Field(default=None, description="Output dimensions [width, height] in pixels")
    num_results: Optional[int] = Field(default=None, description="Number of result images to generate (1-4)")
    seed: Optional[int] = Field(default=None, description="Seed for reproducible results")
    content_moderation: Optional[bool] = Field(default=None, description="Apply content moderation to input and output images")
    force_rmbg: Optional[bool] = Field(default=None, description="Force background removal even if image has alpha channel")


class AppOutput(BaseAppOutput):
    image: File = Field(description="Product placed in lifestyle scene")


class App(BaseApp):
    async def setup(self, config):
        self.client = bria_helper.get_client()
        logger.info("Bria Product Lifestyle (Text) ready")

    async def run(self, input_data: AppInput) -> AppOutput:
        payload = {
            "image_url": input_data.image.uri,
            "scene_description": input_data.scene_description,
        }

        for key in ("optimize_description", "placement_type", "manual_padding", "shot_size", "num_results", "seed", "content_moderation", "force_rmbg"):
            val = getattr(input_data, key)
            if val is not None:
                payload[key] = val

        logger.info("Requesting lifestyle shot by text")
        result = await bria_helper.call_endpoint(self.client, "lifestyle_shot_by_text", payload, base_url=PRODUCT_BASE)

        # v1 returns {"result": [[url, seed, id]]} or {"result": [url, seed, id]}
        r = result["result"]
        if isinstance(r, list) and len(r) > 0 and isinstance(r[0], list):
            r = r[0]
        image_url = r[0] if isinstance(r, list) else r.get("image_url", r)
        # Download with clean client (no auth headers — CloudFront rejects them)
        import httpx as _httpx
        import tempfile
        async with _httpx.AsyncClient(timeout=120) as dl:
            resp = await dl.get(image_url)
            resp.raise_for_status()
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                f.write(resp.content)
                path = f.name
        logger.info(f"Downloaded lifestyle image to {path}")

        with Image.open(path) as img:
            width, height = img.size

        return AppOutput(
            image=File(path=path),
            output_meta=OutputMeta(outputs=[ImageMeta(width=width, height=height, count=1)]),
        )

    async def unload(self):
        await self.client.aclose()

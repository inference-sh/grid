from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field
from typing import Optional, List
from PIL import Image
import logging
import asyncio
import tempfile

from . import bria_helper

logger = logging.getLogger(__name__)

ADS_BASE_URL = "https://engine.prod.bria-api.com/v1"
POLL_INTERVAL = 2.0
MAX_POLL_TIME = 300


class SmartImageScene(BaseAppInput):
    operation: str = Field(
        description="Background operation: 'expand_image' or 'lifestyle_shot_by_text'"
    )
    input: Optional[str] = Field(
        default=None,
        description="Background prompt or hex color. Ignored when operation is 'expand_image'.",
    )


class SmartImage(BaseAppInput):
    input_image_url: str = Field(description="URL of the image to embed within the smart image")
    scene: SmartImageScene = Field(description="Background operation configuration")


class Element(BaseAppInput):
    id: Optional[str] = Field(default=None, description="Unique identifier for the element")
    layer_type: str = Field(description="Element type: 'text' or 'image'")
    content_type: str = Field(
        description="Format such as 'Heading #1', 'Body #1', 'Image #1' (numbers #1-#5)"
    )
    content: str = Field(
        description="Text string or image URL. Empty string to ignore this element."
    )


class SceneResult(BaseAppInput):
    id: Optional[str] = Field(default=None, description="Generated scene ID")
    name: Optional[str] = Field(default=None, description="Scene name from template")
    width: Optional[int] = Field(default=None, description="Image width in pixels")
    height: Optional[int] = Field(default=None, description="Image height in pixels")
    image: File = Field(description="Generated ad image")
    editor_iframe: Optional[str] = Field(
        default=None, description="URL to open the ad in Bria's editor (for iframe embedding)"
    )


class AppInput(BaseAppInput):
    template_id: str = Field(description="Template ID for generating ad scenes (e.g. '1061', '1062')")
    brand_id: Optional[str] = Field(
        default=None,
        description="Brand ID for branded templates. Required for template 1062, ignored for 1061.",
    )
    smart_image: Optional[SmartImage] = Field(
        default=None,
        description="Smart image config for embedding objects/products with AI backgrounds",
    )
    elements: Optional[List[Element]] = Field(
        default=None,
        description="List of text and image elements to customize in the template",
    )
    content_moderation: Optional[bool] = Field(
        default=None,
        description="Enable content moderation on inputs and outputs",
    )


class AppOutput(BaseAppOutput):
    scenes: List[SceneResult] = Field(description="Generated ad scenes")


class App(BaseApp):
    async def setup(self, config):
        self.client = bria_helper.get_client(timeout=360)
        logger.info("Bria Ads Generate ready")

    async def run(self, input_data: AppInput) -> AppOutput:
        payload: dict = {"template_id": input_data.template_id}

        if input_data.brand_id is not None:
            payload["brand_id"] = input_data.brand_id

        if input_data.content_moderation is not None:
            payload["content_moderation"] = input_data.content_moderation

        if input_data.smart_image is not None:
            si = input_data.smart_image
            smart_image_payload: dict = {
                "input_image_url": si.input_image_url,
                "scene": {"operation": si.scene.operation},
            }
            if si.scene.input is not None:
                smart_image_payload["scene"]["input"] = si.scene.input
            payload["smart_image"] = smart_image_payload

        if input_data.elements is not None:
            elements_payload = []
            for el in input_data.elements:
                el_dict: dict = {
                    "layer_type": el.layer_type,
                    "content_type": el.content_type,
                    "content": el.content,
                }
                if el.id is not None:
                    el_dict["id"] = el.id
                elements_payload.append(el_dict)
            payload["elements"] = elements_payload

        logger.info("Requesting ad generation with template %s", input_data.template_id)

        # The ads endpoint is at /v1/ads/generate (not /v2/image/edit).
        # Post directly instead of using the helper's call_endpoint to handle
        # the ads-specific async pattern where scene URLs are returned
        # immediately and become available after background generation.
        url = f"{ADS_BASE_URL}/ads/generate"
        resp = await self.client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()

        # Handle standard async polling if status_url is present
        status_url = data.get("status_url")
        if status_url:
            request_id = data.get("request_id", "unknown")
            logger.info("Polling request %s", request_id)
            elapsed = 0.0
            while elapsed < MAX_POLL_TIME:
                await asyncio.sleep(POLL_INTERVAL)
                elapsed += POLL_INTERVAL
                poll_resp = await self.client.get(status_url)
                poll_resp.raise_for_status()
                poll_data = poll_resp.json()
                status = poll_data.get("status", "")
                if status == "COMPLETED":
                    logger.info("Request %s completed in %.1fs", request_id, elapsed)
                    data = poll_data
                    break
                elif status in ("ERROR", "UNKNOWN"):
                    error = poll_data.get("error", {})
                    raise RuntimeError(
                        f"Bria request {request_id} failed: {error.get('message', status)}"
                    )
            else:
                raise TimeoutError(f"Bria request {request_id} timed out after {MAX_POLL_TIME}s")

        results = data.get("result", [])
        if not results:
            raise RuntimeError("No scenes returned from Bria ads generation")

        scenes: list[SceneResult] = []
        image_metas: list[ImageMeta] = []

        for scene_data in results:
            scene_url = scene_data.get("url", "")
            if not scene_url:
                logger.warning("Scene %s has no URL, skipping", scene_data.get("name"))
                continue

            # Poll the scene URL until the image is ready (non-zero bytes)
            path = await self._download_scene(scene_url)
            if path is None:
                logger.warning("Scene %s produced empty image, skipping", scene_data.get("name"))
                continue

            resolution = scene_data.get("resolution", {})
            width = resolution.get("width")
            height = resolution.get("height")

            if not width or not height:
                try:
                    with Image.open(path) as img:
                        width, height = img.size
                except Exception:
                    width, height = 0, 0

            scenes.append(
                SceneResult(
                    id=scene_data.get("id"),
                    name=scene_data.get("name"),
                    width=width,
                    height=height,
                    image=File(path=path),
                    editor_iframe=scene_data.get("editor_iframe"),
                )
            )
            image_metas.append(ImageMeta(width=width, height=height, count=1))

        logger.info("Generated %d ad scenes", len(scenes))

        return AppOutput(
            scenes=scenes,
            output_meta=OutputMeta(outputs=image_metas),
        )

    async def _download_scene(self, url: str, retries: int = 60) -> str | None:
        """Download a scene image, retrying until content is available."""
        for attempt in range(retries):
            try:
                resp = await self.client.get(url)
                if resp.status_code == 200 and len(resp.content) > 0:
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                        f.write(resp.content)
                        return f.name
            except Exception as e:
                logger.debug("Scene download attempt %d failed: %s", attempt + 1, e)
            await asyncio.sleep(POLL_INTERVAL)
        return None

    async def unload(self):
        await self.client.aclose()

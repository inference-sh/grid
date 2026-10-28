"""Meshy 7.1: text, image or multi-image to 3D on Meshy's own API (https://api.meshy.ai/openapi).

The endpoint follows the inputs:
- no images: Text to 3D, a `preview` task (mesh) and, when textured, a `refine` task on top of it
- one image: Image to 3D
- two to four images: Multi-Image to 3D, the first image is the front view

Meshy bills in credits and reports them per task as `consumed_credits`; the
output reports their sum in `output_meta`.
"""

import asyncio
import base64
import logging
import os
import tempfile
import time
from typing import List, Literal, Optional

import httpx
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, RawMeta, function
from inferencesh.images import ImageLimits, fit_image
from pydantic import Field, model_validator

BASE_URL = "https://api.meshy.ai/openapi"
API_KEY_SECRET = "MESHY_API_KEY"
AI_MODEL = "meshy-7.1"

MAX_RETRIES = 3
RETRY_STATUSES = {429, 500, 502, 503, 504}
# Meshy asks clients to wait at least 60 minutes from creation before giving up on a task.
TASK_TIMEOUT = 60 * 60
POLL_INTERVAL = 5

# Meshy takes jpg, png and webp up to 20 MB per image, as a URL or a data URI.
IMAGE_LIMITS = ImageLimits(max_bytes=20 * 1024 * 1024, mime_types=("image/png", "image/jpeg", "image/webp"))
MAX_IMAGES = 4
MAX_PROMPT_CHARS = 800

ModelFormat = Literal["glb", "fbx", "obj", "stl", "usdz", "3mf"]


class UpstreamError(RuntimeError):
    """Meshy answered with an error, or a task failed. The message carries Meshy's reason."""


class AppInput(BaseAppInput):
    prompt: Optional[str] = Field(
        default=None,
        max_length=MAX_PROMPT_CHARS,
        description="What to make, for Text to 3D. Used when no images are given.",
        examples=["a weathered bronze statue of a fox sitting upright"],
    )
    images: List[File] = Field(
        default_factory=list,
        max_length=MAX_IMAGES,
        description="One image of the object, or two to four views of it with the front view first. jpg, png or webp.",
    )
    should_texture: bool = Field(default=True, description="Texture the mesh. Off returns the bare mesh and costs less.")
    enable_pbr: bool = Field(default=False, description="Add metallic, roughness and normal maps. Needs should_texture.")
    texture_resolution: Literal["2k", "4k", "8k"] = Field(default="2k", description="Texture size. 8k costs more.")
    texture_prompt: Optional[str] = Field(
        default=None, max_length=MAX_PROMPT_CHARS, description="Text that guides the texture. Takes precedence over texture_image."
    )
    texture_image: Optional[File] = Field(default=None, description="An image whose look the texture should follow.")
    geometry_resolution: Literal["standard", "2k", "4k"] = Field(
        default="standard", description="Mesh detail. 2k and 4k cost 5 credits more; 4k is not available with several images."
    )
    topology: Literal["triangle", "quad"] = Field(default="triangle", description="Polygon type of the mesh.")
    should_remesh: bool = Field(default=False, description="Remesh to target_polycount and topology.")
    target_polycount: Optional[int] = Field(
        default=None, ge=100, le=300000, description="Polygon count when remeshing, 100 to 300000. Empty: 30000."
    )
    pose_mode: Literal["", "a-pose", "t-pose"] = Field(default="", description="Put a character in an A or T pose. Empty: as given.")
    image_enhancement: bool = Field(default=True, description="Let Meshy clean up the input images first.")
    auto_size: bool = Field(default=False, description="Scale the model to a real-world size.")
    origin_at: Literal["bottom", "center"] = Field(default="bottom", description="Where the origin sits with auto_size.")
    format: ModelFormat = Field(default="glb", description="File format of the model.")
    moderation: bool = Field(default=False, description="Have Meshy refuse harmful prompts and images.")

    @model_validator(mode="after")
    def check_inputs(self):
        if not self.images and not (self.prompt or "").strip():
            raise ValueError("give a prompt or at least one image")
        if self.images and (self.prompt or "").strip():
            raise ValueError("prompt is for Text to 3D; with images, use texture_prompt to guide the texture")
        if self.enable_pbr and not self.should_texture:
            raise ValueError("enable_pbr needs should_texture")
        if len(self.images) > 1 and self.geometry_resolution == "4k":
            raise ValueError("geometry_resolution 4k is not available with several images; use 2k")
        return self


class AppOutput(BaseAppOutput):
    model: File = Field(description="The 3D model.")
    thumbnail: Optional[File] = Field(default=None, description="A preview render of the model.")
    textures: List[File] = Field(default_factory=list, description="Texture maps: base color, and metallic, roughness and normal with PBR.")


def data_uri(path: str) -> str:
    data, mime = fit_image(path, IMAGE_LIMITS)
    return "data:%s;base64,%s" % (mime, base64.b64encode(data).decode("ascii"))


class App(BaseApp):
    async def setup(self, config):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        api_key = os.environ.get(API_KEY_SECRET)
        if not api_key:
            raise ValueError(f"{API_KEY_SECRET} is not set: connect the Meshy credential")
        self.client = httpx.AsyncClient(
            base_url=BASE_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(120, connect=15),
        )
        self.task_path: Optional[str] = None

    async def _request(self, method: str, path: str, retry: bool = True, **kwargs) -> dict:
        """One API call: 429 and 5xx retried when `retry`, raises on any other error."""
        for attempt in range(MAX_RETRIES + 1):
            response = await self.client.request(method, path, **kwargs)
            if response.status_code in RETRY_STATUSES and retry and attempt < MAX_RETRIES:
                delay = float(response.headers.get("Retry-After", 2 ** (attempt + 1)))
                self.logger.warning("%s %s: %d, retrying in %.0fs", method, path, response.status_code, delay)
                await asyncio.sleep(delay)
                continue
            if response.is_error:
                raise UpstreamError("Meshy %d: %s" % (response.status_code, response.text[:500]))
            return response.json()

    async def _create(self, path: str, body: dict) -> str:
        # Creates carry no idempotency key: only a 429 is safe to resend, since no task was made.
        for attempt in range(MAX_RETRIES + 1):
            response = await self.client.post(path, json=body)
            if response.status_code == 429 and attempt < MAX_RETRIES:
                delay = float(response.headers.get("Retry-After", 10 * (attempt + 1)))
                self.logger.warning("create %s: 429 %s, retrying in %.0fs", path, response.text[:200], delay)
                await asyncio.sleep(delay)
                continue
            if response.is_error:
                raise UpstreamError("Meshy %d: %s" % (response.status_code, response.text[:500]))
            task_id = response.json()["result"]
            self.task_path = "%s/%s" % (path, task_id)
            self.logger.info("created %s task %s", path, task_id)
            return task_id

    async def _wait(self, path: str, task_id: str) -> dict:
        started = time.monotonic()
        last_progress = -1
        while time.monotonic() - started < TASK_TIMEOUT:
            task = await self._request("GET", "%s/%s" % (path, task_id))
            status = task.get("status")
            if status == "SUCCEEDED":
                self.logger.info("task %s succeeded, %s credits", task_id, task.get("consumed_credits"))
                return task
            if status in ("FAILED", "CANCELED"):
                error = task.get("task_error") or {}
                raise UpstreamError("Meshy task %s %s: %s" % (task_id, status.lower(), error.get("message") or error))
            progress = task.get("progress", 0)
            if progress != last_progress:
                self.logger.info("task %s: %s %d%%", task_id, status, progress)
                last_progress = progress
            await asyncio.sleep(POLL_INTERVAL)
        raise UpstreamError("Meshy task %s did not finish in %d minutes" % (task_id, TASK_TIMEOUT // 60))

    def _common(self, input_data: AppInput) -> dict:
        body = {
            "ai_model": AI_MODEL,
            "target_formats": [input_data.format],
            "moderation": input_data.moderation,
            "auto_size": input_data.auto_size,
            "origin_at": input_data.origin_at,
        }
        return body

    def _geometry(self, input_data: AppInput) -> dict:
        body = {
            "geometry_resolution": input_data.geometry_resolution,
            "topology": input_data.topology,
            "should_remesh": input_data.should_remesh,
            "pose_mode": input_data.pose_mode,
        }
        if input_data.target_polycount is not None:
            body["target_polycount"] = input_data.target_polycount
        return body

    def _texture(self, input_data: AppInput) -> dict:
        body = {"enable_pbr": input_data.enable_pbr, "texture_resolution": input_data.texture_resolution}
        if input_data.texture_prompt:
            body["texture_prompt"] = input_data.texture_prompt
        elif input_data.texture_image is not None:
            body["texture_image_url"] = data_uri(input_data.texture_image.path)
        return body

    def _image_body(self, input_data: AppInput) -> dict:
        body = {**self._common(input_data), **self._geometry(input_data)}
        body["should_texture"] = input_data.should_texture
        body["image_enhancement"] = input_data.image_enhancement
        if input_data.should_texture:
            body.update(self._texture(input_data))
        if len(input_data.images) == 1:
            body["image_url"] = data_uri(input_data.images[0].path)
        else:
            body["image_urls"] = [data_uri(image.path) for image in input_data.images]
        return body

    async def _download(self, url: str, suffix: str) -> str:
        fd, out_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        async with httpx.AsyncClient(timeout=httpx.Timeout(300, connect=15), follow_redirects=True) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                with open(out_path, "wb") as f:
                    async for chunk in response.aiter_bytes():
                        f.write(chunk)
        return out_path

    @function(annotations={"title": "Generate 3D model", "readOnlyHint": False, "destructiveHint": False, "openWorldHint": True})
    async def run(self, input_data: AppInput) -> AppOutput:
        """Make a 3D model from a text prompt, one image, or up to four views of an object."""
        credits = 0
        if not input_data.images:
            mode = "text"
            path = "/v2/text-to-3d"
            body = {**self._common(input_data), **self._geometry(input_data), "mode": "preview", "prompt": input_data.prompt.strip()}
            self.logger.info("text to 3D preview: %d chars, geometry %s", len(body["prompt"]), input_data.geometry_resolution)
            task = await self._wait(path, await self._create(path, body))
            credits += int(task.get("consumed_credits") or 0)
            if input_data.should_texture:
                body = {**self._common(input_data), **self._texture(input_data), "mode": "refine", "preview_task_id": task["id"]}
                self.logger.info("text to 3D refine: texture %s, pbr %s", input_data.texture_resolution, input_data.enable_pbr)
                task = await self._wait(path, await self._create(path, body))
                credits += int(task.get("consumed_credits") or 0)
        else:
            mode = "image" if len(input_data.images) == 1 else "multi-image"
            path = "/v1/image-to-3d" if mode == "image" else "/v1/multi-image-to-3d"
            body = self._image_body(input_data)
            self.logger.info("%s to 3D: %d image(s), texture %s", mode, len(input_data.images), input_data.should_texture)
            task = await self._wait(path, await self._create(path, body))
            credits += int(task.get("consumed_credits") or 0)
        self.task_path = None

        model_url = (task.get("model_urls") or {}).get(input_data.format)
        if not model_url:
            raise UpstreamError("Meshy returned no %s model: %s" % (input_data.format, list((task.get("model_urls") or {}).keys())))
        model_path = await self._download(model_url, "." + input_data.format)

        thumbnail = None
        if task.get("thumbnail_url"):
            thumbnail = File(path=await self._download(task["thumbnail_url"], ".png"))
        textures = []
        for maps in task.get("texture_urls") or []:
            for name in ("base_color", "metallic", "roughness", "normal", "emission"):
                if maps.get(name):
                    textures.append(File(path=await self._download(maps[name], "_%s.png" % name)))

        self.logger.info("usage: credits=%d mode=%s model=%s", credits, mode, AI_MODEL)
        meta = RawMeta(
            extra={
                "credits": credits,
                "mode": mode,
                "model": AI_MODEL,
                "textured": input_data.should_texture,
                "texture_resolution": input_data.texture_resolution if input_data.should_texture else "",
                "geometry_resolution": input_data.geometry_resolution,
            }
        )
        return AppOutput(
            model=File(path=model_path),
            thumbnail=thumbnail,
            textures=textures,
            output_meta=OutputMeta(inputs=[], outputs=[meta]),
        )

    async def on_cancel(self):
        # Meshy refunds a task deleted while PENDING; a running task answers 409 and finishes.
        if self.task_path:
            try:
                await self.client.delete(self.task_path)
            except Exception as e:
                self.logger.warning("cancel %s: %s", self.task_path, e)
        return True

    async def unload(self):
        await self.client.aclose()

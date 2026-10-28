"""Rodin Gen-2.5: text or image to 3D on Hyper3D's own API (https://api.hyper3d.com/api/v2, by Deemos).

One submit (`POST /rodin`, multipart) takes a prompt, or one to five images of
one object. The task is polled with its subscription key (`POST /status`) and
its files are listed with the task uuid (`POST /download`).

Hyper3D bills in credits and reports them per submit as `consumed`; the output
reports them in `output_meta`.
"""

import asyncio
import logging
import os
import tempfile
import time
from typing import List, Literal, Optional

import httpx
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, RawMeta, function
from inferencesh.images import ImageLimits, fit_image
from pydantic import Field, model_validator

BASE_URL = "https://api.hyper3d.com/api/v2"
API_KEY_SECRET = "HYPER3D_API_KEY"

MAX_RETRIES = 3
RETRY_STATUSES = {429, 500, 502, 503, 504}
TASK_TIMEOUT = 30 * 60
# Hyper3D asks for a first check after 5 s, backing off to at most 30 s.
POLL_INTERVALS = (5, 5, 10, 10, 15, 20, 30)

IMAGE_LIMITS = ImageLimits(max_bytes=10 * 1024 * 1024, mime_types=("image/png", "image/jpeg", "image/webp"))
EXTENSIONS = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}

Tier = Literal["Gen-2.5-Extreme-Low", "Gen-2.5-Low", "Gen-2.5-Medium", "Gen-2.5-High", "Gen-2.5-Extreme-High"]
ModelFormat = Literal["glb", "usdz", "fbx", "obj", "stl"]


class UpstreamError(RuntimeError):
    """Hyper3D answered with an error, or a task failed. The message carries Hyper3D's code."""


class AppInput(BaseAppInput):
    prompt: Optional[str] = Field(
        default=None,
        max_length=1024,
        description="What to make, for text to 3D. With images, leave it empty unless the images need explaining.",
        examples=["a carved wooden owl perched on a branch"],
    )
    images: List[File] = Field(
        default_factory=list,
        max_length=5,
        description="One to five images of one object. The first image sets the materials.",
    )
    tier: Tier = Field(
        default="Gen-2.5-Medium",
        description="Quality tier. Extreme-High costs 0.5 credits more and adds a mesh of up to 10M faces.",
    )
    mesh_mode: Literal["Raw", "Quad"] = Field(default="Raw", description="Raw triangles, or a quad mesh.")
    quality: Literal["high", "medium", "low", "extra-low"] = Field(
        default="medium",
        description="Face count. Raw: 1M, 500K, 60K, 20K. Quad: 50K, 18K, 8K, 4K.",
    )
    quality_override: Optional[int] = Field(
        default=None,
        ge=500,
        le=2000000,
        description="Exact face count, instead of quality. Above 1M needs the High or Extreme-High tier; Quad takes at most 200K.",
    )
    geometry_file_format: ModelFormat = Field(default="glb", description="File format of the model.")
    material: Literal["PBR", "Shaded", "All", "Hybrid", "None"] = Field(
        default="PBR", description="PBR maps, baked shading, both, hybrid, or no material."
    )
    texture_mode: Optional[Literal["legacy", "extreme-low", "low", "medium", "high", "extreme-high"]] = Field(
        default=None,
        description="Texture quality. Empty: follows the tier. extreme-high gives 6K textures and costs 2 credits more.",
    )
    hd_texture: bool = Field(default=False, description="Higher quality texture.")
    uhd_texture: bool = Field(default=False, description="4K textures, 8K with high_pack.")
    high_pack: bool = Field(default=False, description="The HighPack add-on: 4K textures, 8K with uhd_texture.")
    texture_delight: bool = Field(default=False, description="Remove lighting baked into the images from the texture.")
    geometry_instruct_mode: Literal["creative", "faithful"] = Field(
        default="creative", description="faithful follows the images more literally (experimental)."
    )
    is_symmetric: Literal["unknown", "symmetric", "balanced", "asymmetric"] = Field(
        default="unknown", description="Symmetry of the object, if known."
    )
    is_micro: bool = Field(default=False, description="Micro detail. Extreme-High tier only.")
    t_a_pose: bool = Field(default=False, description="Put a human-like subject in a T or A pose.")
    use_original_alpha: bool = Field(default=False, description="Use the images' own transparency as the mask.")
    seed: Optional[int] = Field(default=None, ge=0, le=65535, description="Seed, 0 to 65535.")

    @model_validator(mode="after")
    def check_inputs(self):
        if not self.images and not (self.prompt or "").strip():
            raise ValueError("give a prompt or at least one image")
        if self.is_micro and self.tier != "Gen-2.5-Extreme-High":
            raise ValueError("is_micro needs the Gen-2.5-Extreme-High tier")
        if self.quality_override is not None:
            if self.mesh_mode == "Quad" and self.quality_override > 200000:
                raise ValueError("Quad meshes take at most 200000 faces")
            if self.quality_override > 1000000 and self.tier not in ("Gen-2.5-High", "Gen-2.5-Extreme-High"):
                raise ValueError("more than 1000000 faces needs the Gen-2.5-High or Gen-2.5-Extreme-High tier")
        return self


class AppOutput(BaseAppOutput):
    model: File = Field(description="The 3D model.")
    preview: Optional[File] = Field(default=None, description="A preview render of the model.")
    files: List[File] = Field(default_factory=list, description="Every other file Hyper3D returned: textures, and the high-poly mesh of Extreme-High.")


def form_bool(value: bool) -> str:
    return "true" if value else "false"


class App(BaseApp):
    async def setup(self, config):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        api_key = os.environ.get(API_KEY_SECRET)
        if not api_key:
            raise ValueError(f"{API_KEY_SECRET} is not set: connect the Hyper3D credential")
        self.client = httpx.AsyncClient(
            base_url=BASE_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(300, connect=15),
        )

    async def _request(self, path: str, retry: bool = True, **kwargs) -> dict:
        """One POST: 429 and 5xx retried when `retry`. Hyper3D reports app errors as `error` in a 2xx body."""
        for attempt in range(MAX_RETRIES + 1):
            response = await self.client.post(path, **kwargs)
            if response.status_code in RETRY_STATUSES and retry and attempt < MAX_RETRIES:
                delay = float(response.headers.get("Retry-After", 2 ** (attempt + 1)))
                self.logger.warning("%s: %d, retrying in %.0fs", path, response.status_code, delay)
                await asyncio.sleep(delay)
                continue
            if response.is_error:
                raise UpstreamError("Hyper3D %d: %s" % (response.status_code, response.text[:500]))
            body = response.json()
            if body.get("error"):
                raise UpstreamError("Hyper3D %s: %s" % (body["error"], body.get("message", "")))
            return body

    def _form(self, input_data: AppInput) -> dict:
        data = {
            "tier": input_data.tier,
            "mesh_mode": input_data.mesh_mode,
            "quality": input_data.quality,
            "geometry_file_format": input_data.geometry_file_format,
            "material": input_data.material,
            "hd_texture": form_bool(input_data.hd_texture),
            "uhd_texture": form_bool(input_data.uhd_texture),
            "texture_delight": form_bool(input_data.texture_delight),
            "geometry_instruct_mode": input_data.geometry_instruct_mode,
            "is_symmetric": input_data.is_symmetric,
            "is_micro": form_bool(input_data.is_micro),
            "TAPose": form_bool(input_data.t_a_pose),
            "use_original_alpha": form_bool(input_data.use_original_alpha),
            "preview_render": "true",
        }
        if (input_data.prompt or "").strip():
            data["prompt"] = input_data.prompt.strip()
        if input_data.quality_override is not None:
            data["quality_override"] = str(input_data.quality_override)
        if input_data.texture_mode:
            data["texture_mode"] = input_data.texture_mode
        if input_data.high_pack:
            data["addons"] = "HighPack"
        if input_data.seed is not None:
            data["seed"] = str(input_data.seed)
        return data

    def _images(self, input_data: AppInput) -> list:
        parts = []
        for i, image in enumerate(input_data.images):
            data, mime = fit_image(image.path, IMAGE_LIMITS)
            parts.append(("images", ("image_%d%s" % (i, EXTENSIONS.get(mime, ".png")), data, mime)))
        return parts

    async def _wait(self, subscription_key: str) -> None:
        started = time.monotonic()
        last = None
        for attempt in range(10**6):
            await asyncio.sleep(POLL_INTERVALS[min(attempt, len(POLL_INTERVALS) - 1)])
            jobs = (await self._request("/status", json={"subscription_key": subscription_key})).get("jobs") or []
            statuses = [job.get("status") for job in jobs]
            if statuses != last:
                self.logger.info("jobs: %s", ", ".join(statuses) or "none yet")
                last = statuses
            if "Failed" in statuses:
                raise UpstreamError("Hyper3D generation failed: %s" % jobs)
            if statuses and all(s == "Done" for s in statuses):
                return
            if time.monotonic() - started > TASK_TIMEOUT:
                break
        raise UpstreamError("Hyper3D task did not finish in %d minutes" % (TASK_TIMEOUT // 60))

    async def _download(self, url: str, name: str) -> str:
        fd, out_path = tempfile.mkstemp(suffix="_" + os.path.basename(name))
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
        """Make a 3D model from a text prompt, or from one to five images of an object."""
        mode = "image" if input_data.images else "text"
        self.logger.info(
            "%s to 3D: %d image(s), tier %s, %s %s, format %s",
            mode, len(input_data.images), input_data.tier, input_data.mesh_mode, input_data.quality, input_data.geometry_file_format,
        )
        # Always multipart, also without images: httpx would send fields alone urlencoded.
        parts = [(key, (None, value)) for key, value in self._form(input_data).items()] + self._images(input_data)
        # A submit has no idempotency key: only retry it when nothing was accepted (429).
        submitted = None
        for attempt in range(MAX_RETRIES + 1):
            response = await self.client.post("/rodin", files=parts)
            if response.status_code == 429 and attempt < MAX_RETRIES:
                delay = float(response.headers.get("Retry-After", 10 * (attempt + 1)))
                self.logger.warning("submit: 429, retrying in %.0fs", delay)
                await asyncio.sleep(delay)
                continue
            if response.is_error:
                raise UpstreamError("Hyper3D %d: %s" % (response.status_code, response.text[:500]))
            submitted = response.json()
            break
        if submitted.get("error") or not submitted.get("uuid"):
            raise UpstreamError("Hyper3D %s: %s" % (submitted.get("error"), submitted.get("message", "")))
        task_uuid = submitted["uuid"]
        credits = float(submitted.get("consumed") or 0)
        self.logger.info("submitted task %s, %s credits", task_uuid, credits)

        await self._wait(submitted["jobs"]["subscription_key"])
        listing = (await self._request("/download", json={"task_uuid": task_uuid})).get("list") or []
        self.logger.info("files: %s", ", ".join(item.get("name", "?") for item in listing))

        model_path, preview_path, others = None, None, []
        ext = "." + input_data.geometry_file_format
        for item in listing:
            name = item.get("name") or "file"
            path = await self._download(item["url"], name)
            lower = name.lower()
            if model_path is None and lower.endswith(ext) and "high" not in lower:
                model_path = path
            elif preview_path is None and "preview" in lower:
                preview_path = path
            else:
                others.append(File(path=path))
        if model_path is None:
            # The high-poly mesh of Extreme-High is the only geometry file: use it.
            match = [f for f in others if f.path.lower().endswith(ext)]
            if not match:
                raise UpstreamError("Hyper3D returned no %s model: %s" % (ext, [i.get("name") for i in listing]))
            model_path = match[0].path
            others = [f for f in others if f.path != model_path]

        self.logger.info("usage: credits=%s mode=%s tier=%s", credits, mode, input_data.tier)
        meta = RawMeta(
            extra={
                "credits": credits,
                "mode": mode,
                "tier": input_data.tier,
                "texture_mode": input_data.texture_mode or "",
                "high_pack": input_data.high_pack,
            }
        )
        return AppOutput(
            model=File(path=model_path),
            preview=File(path=preview_path) if preview_path else None,
            files=others,
            output_meta=OutputMeta(inputs=[], outputs=[meta]),
        )

    async def unload(self):
        await self.client.aclose()

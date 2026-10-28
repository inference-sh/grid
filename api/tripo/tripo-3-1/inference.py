"""Tripo 3.1: text, image or multiview to 3D on Tripo's own API (https://openapi.tripo3d.ai/v3, H series model v3.1-20260211).

The endpoint follows the inputs:
- no images: `text-to-model`
- a front image only: `image-to-model`
- a front image and one to three of left, back, right: `multiview-to-model`

Images are uploaded first (`POST /v3/files`) and referenced by their file token;
the API calls are in tripo_helper.py.

Tripo bills in credits ($0.01 each) and reports them per task as
`credits_consumed`; the output reports them in `output_meta`.
"""

import logging
from typing import Literal, Optional

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, RawMeta, function
from inferencesh.images import ImageLimits, fit_image
from pydantic import Field, model_validator

from .tripo_helper import TripoClient, UpstreamError, credits, download, output_url

MODEL = "v3.1-20260211"
# texture_quality "fast" exists only on this texture model.
FAST_TEXTURE_VERSION = "v3.5-20260815"

# The upload endpoint takes JPEG and PNG up to 20 MB.
IMAGE_LIMITS = ImageLimits(max_bytes=20 * 1024 * 1024, mime_types=("image/png", "image/jpeg"))
EXTENSIONS = {"image/png": ".png", "image/jpeg": ".jpg"}


class AppInput(BaseAppInput):
    prompt: Optional[str] = Field(
        default=None,
        max_length=1024,
        description="What to make, for text to 3D. Used when no image is given.",
        examples=["a stylized ceramic teapot shaped like a snail"],
    )
    negative_prompt: Optional[str] = Field(default=None, max_length=255, description="What to keep out, for text to 3D.")
    image: Optional[File] = Field(default=None, description="The object, or its front view when side views are given. JPEG or PNG.")
    left_image: Optional[File] = Field(default=None, description="Left view of the object, for multiview. Needs image.")
    back_image: Optional[File] = Field(default=None, description="Back view of the object, for multiview. Needs image.")
    right_image: Optional[File] = Field(default=None, description="Right view of the object, for multiview. Needs image.")
    texture: bool = Field(default=True, description="Texture the model. Off returns the bare mesh and costs less.")
    pbr: bool = Field(default=True, description="Add PBR material maps. Turns texture on.")
    texture_quality: Literal["fast", "standard", "detailed", "extreme"] = Field(
        default="standard", description="fast and standard cost the same; detailed (HD) and extreme (8K) cost more."
    )
    geometry_quality: Literal["standard", "detailed"] = Field(default="standard", description="detailed is Ultra geometry and costs 20 credits more.")
    face_limit: Optional[int] = Field(
        default=None,
        ge=500,
        le=2000000,
        description="Maximum face count. Empty: chosen by Tripo. Up to 1.5M, 2M with detailed geometry, 150K with quad, 20K with smart_low_poly.",
    )
    quad: bool = Field(default=False, description="Quad mesh, delivered as FBX. Costs 5 credits more.")
    smart_low_poly: bool = Field(default=False, description="Hand-crafted style low-poly topology. Costs 10 credits more.")
    generate_parts: bool = Field(default=False, description="Split the model into editable parts. Needs texture and pbr off; costs 20 credits more.")
    auto_size: bool = Field(default=False, description="Scale the model to real-world size in meters.")
    enable_image_autofix: bool = Field(default=False, description="Let Tripo clean up a single input image first.")
    texture_alignment: Literal["original_image", "geometry"] = Field(
        default="original_image", description="Align the texture to the input image, or to the geometry."
    )
    orientation: Literal["default", "align_image"] = Field(default="default", description="align_image turns the model to match the image.")
    model_seed: Optional[int] = Field(default=None, ge=0, description="Seed for the geometry.")
    texture_seed: Optional[int] = Field(default=None, ge=0, description="Seed for the texture.")
    image_seed: Optional[int] = Field(default=None, ge=0, description="Seed for the image Tripo draws from the prompt, for text to 3D.")

    @model_validator(mode="after")
    def check_inputs(self):
        views = [self.left_image, self.back_image, self.right_image]
        if self.image is None and any(v is not None for v in views):
            raise ValueError("side views need image as the front view")
        if self.image is None and not (self.prompt or "").strip():
            raise ValueError("give a prompt or an image")
        if self.image is not None and (self.prompt or "").strip():
            raise ValueError("prompt is for text to 3D; leave it empty with an image")
        if self.generate_parts and (self.texture or self.pbr):
            raise ValueError("generate_parts needs texture and pbr off")
        if self.quad and self.face_limit is not None and self.face_limit > 150000:
            raise ValueError("quad meshes take at most 150000 faces")
        if self.smart_low_poly and self.face_limit is not None and self.face_limit > (10000 if self.quad else 20000):
            raise ValueError("smart_low_poly takes at most 20000 triangles or 10000 quads")
        if self.face_limit is not None and self.face_limit > 1500000 and self.geometry_quality != "detailed":
            raise ValueError("face_limit above 1500000 needs geometry_quality detailed")
        return self


class AppOutput(BaseAppOutput):
    model: File = Field(description="The 3D model: GLB, or FBX for a quad mesh.")
    rendered_image: Optional[File] = Field(default=None, description="A preview render of the model.")


class App(BaseApp):
    async def setup(self, config):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        self.tripo = TripoClient()

    async def _upload(self, file: File) -> str:
        data, mime = fit_image(file.path, IMAGE_LIMITS)
        return await self.tripo.upload("image" + EXTENSIONS.get(mime, ".png"), data, mime)

    def _body(self, input_data: AppInput) -> dict:
        body = {
            "model": MODEL,
            "texture": input_data.texture or input_data.pbr,
            "pbr": input_data.pbr,
            "geometry_quality": input_data.geometry_quality,
            "auto_size": input_data.auto_size,
            "quad": input_data.quad,
            "smart_low_poly": input_data.smart_low_poly,
            "generate_parts": input_data.generate_parts,
        }
        if body["texture"]:
            body["texture_quality"] = input_data.texture_quality
            if input_data.texture_quality == "fast":
                body["texture_version"] = FAST_TEXTURE_VERSION
        for name in ("face_limit", "model_seed", "texture_seed"):
            value = getattr(input_data, name)
            if value is not None:
                body[name] = value
        return body

    @function(annotations={"title": "Generate 3D model", "readOnlyHint": False, "destructiveHint": False, "openWorldHint": True})
    async def run(self, input_data: AppInput) -> AppOutput:
        """Make a 3D model from a text prompt, an image, or a front view with up to three side views."""
        body = self._body(input_data)
        views = {"left": input_data.left_image, "back": input_data.back_image, "right": input_data.right_image}
        views = {name: f for name, f in views.items() if f is not None}
        if input_data.image is None:
            mode, path = "text", "/generation/text-to-model"
            body["prompt"] = input_data.prompt.strip()
            if input_data.negative_prompt:
                body["negative_prompt"] = input_data.negative_prompt
            if input_data.image_seed is not None:
                body["image_seed"] = input_data.image_seed
        else:
            body["texture_alignment"] = input_data.texture_alignment
            body["orientation"] = input_data.orientation
            front = await self._upload(input_data.image)
            if views:
                mode, path = "multiview", "/generation/multiview-to-model"
                body["inputs"] = [{"front": front}] + [{name: await self._upload(f)} for name, f in views.items()]
            else:
                mode, path = "image", "/generation/image-to-model"
                body["input"] = front
                body["enable_image_autofix"] = input_data.enable_image_autofix
        self.logger.info(
            "%s to 3D: texture %s/%s, geometry %s, quad %s, low poly %s, parts %s",
            mode, body["texture"], body.get("texture_quality", "-"), input_data.geometry_quality,
            input_data.quad, input_data.smart_low_poly, input_data.generate_parts,
        )
        task = await self.tripo.run(path, body)

        output = task.get("output") or {}
        model_url = output_url(task)
        if not model_url:
            raise UpstreamError("Tripo returned no model: %s" % list(output.keys()))
        suffix = ".fbx" if input_data.quad or ".fbx" in model_url.split("?")[0].lower() else ".glb"
        model_path = await download(model_url, suffix)
        rendered = None
        if output.get("rendered_image_url"):
            rendered = File(path=await download(output["rendered_image_url"], ".webp"))

        used = credits(task)
        self.logger.info("usage: credits=%s mode=%s model=%s", used, mode, MODEL)
        meta = RawMeta(
            extra={
                "credits": used,
                "mode": mode,
                "model": MODEL,
                "textured": body["texture"],
                "texture_quality": body.get("texture_quality", ""),
                "geometry_quality": input_data.geometry_quality,
                "quad": input_data.quad,
                "smart_low_poly": input_data.smart_low_poly,
                "generate_parts": input_data.generate_parts,
            }
        )
        return AppOutput(model=File(path=model_path), rendered_image=rendered, output_meta=OutputMeta(inputs=[], outputs=[meta]))

    async def unload(self):
        await self.tripo.close()

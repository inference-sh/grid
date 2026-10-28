"""Hunyuan 3D 3.1: text, image or multiview to 3D on Tencent Cloud International (`hunyuan` service, Pro job, Model 3.1).

`SubmitHunyuanTo3DProJob` takes a prompt, or a front image with optional side
views; `QueryHunyuanTo3DProJob` is polled until the job is DONE (signing and
polling in tencent_helper.py).

Tencent bills Pro jobs in credits by option and does not report them on the
international API, so the app computes them from the request (see pricing.md).
"""

import base64
import logging
from typing import List, Literal, Optional

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, RawMeta, function
from inferencesh.images import ImageLimits, fit_image
from pydantic import Field, model_validator

from .tencent_helper import TencentClient, result_files

MODEL = "3.1"

# Each side 128 to 5000 px; at most 8 MB after base64, so about 6 MB raw, for the image and for all views together.
RAW_BUDGET = 6 * 1024 * 1024
MAX_EDGE = 5000

# Pro credits by GenerateType, and each paid option (https://www.tencentcloud.com/document/product/1284/75281).
BASE_CREDITS = {"Normal": 25, "Geometry": 15}
OPTION_CREDITS = 10

View = Literal["left", "right", "back", "top", "bottom", "left_front", "right_front"]


class ViewImage(BaseAppInput):
    view: View = Field(description="Which side of the object the image shows.")
    image: File = Field(description="The view. JPG or PNG.")


class AppInput(BaseAppInput):
    prompt: Optional[str] = Field(
        default=None,
        max_length=1024,
        description="What to make, for text to 3D. Used when no image is given.",
        examples=["a low wooden stool with three legs"],
    )
    image: Optional[File] = Field(default=None, description="The object, or its front view when views are given. jpg, png or webp.")
    views: List[ViewImage] = Field(
        default_factory=list,
        max_length=7,
        description="Other views of the object, one per side: left, right, back, top, bottom, left_front, right_front. Needs image; costs 10 credits more.",
    )
    generate_type: Literal["Normal", "Geometry"] = Field(
        default="Normal", description="Normal is a textured model; Geometry an untextured GLB and costs less."
    )
    enable_pbr: bool = Field(default=False, description="PBR materials. Costs 10 credits more; not with Geometry.")
    face_count: Optional[int] = Field(
        default=None, ge=3000, le=1500000, description="Face count, 3000 to 1500000. Empty: 500000. Setting it costs 10 credits more."
    )

    @model_validator(mode="after")
    def check_inputs(self):
        if self.image is None and not (self.prompt or "").strip():
            raise ValueError("give a prompt or an image")
        if self.image is not None and (self.prompt or "").strip():
            raise ValueError("prompt is for text to 3D; leave it empty with an image")
        if self.views and self.image is None:
            raise ValueError("views need image as the front view")
        sides = [v.view for v in self.views]
        if len(sides) != len(set(sides)):
            raise ValueError("give each view once")
        if self.enable_pbr and self.generate_type == "Geometry":
            raise ValueError("enable_pbr has no effect with Geometry")
        return self


class AppOutput(BaseAppOutput):
    model: File = Field(description="The 3D model.")
    preview: Optional[File] = Field(default=None, description="A preview render of the model.")
    files: List[File] = Field(default_factory=list, description="Every other file Tencent returned.")


def credits_for(input_data: AppInput) -> int:
    credits = BASE_CREDITS[input_data.generate_type]
    credits += OPTION_CREDITS * sum([bool(input_data.views), input_data.enable_pbr, input_data.face_count is not None])
    return credits


def encode_image(path: str, max_bytes: int, png_jpeg_only: bool) -> str:
    mimes = ("image/png", "image/jpeg") if png_jpeg_only else ("image/png", "image/jpeg", "image/webp")
    data, _ = fit_image(path, ImageLimits(max_long_edge=MAX_EDGE, max_bytes=max_bytes, mime_types=mimes))
    return base64.b64encode(data).decode("ascii")


class App(BaseApp):
    async def setup(self, config):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        self.tencent = TencentClient()

    def _params(self, input_data: AppInput) -> dict:
        params = {"Model": MODEL, "GenerateType": input_data.generate_type}
        if input_data.enable_pbr:
            params["EnablePBR"] = True
        if input_data.face_count is not None:
            params["FaceCount"] = input_data.face_count
        if input_data.image is None:
            params["Prompt"] = input_data.prompt.strip()
            return params
        if not input_data.views:
            params["ImageBase64"] = encode_image(input_data.image.path, RAW_BUDGET, png_jpeg_only=False)
            return params
        # The front image and its views share one 8 MB budget, and views take JPG or PNG only.
        share = RAW_BUDGET // (len(input_data.views) + 1)
        params["ImageBase64"] = encode_image(input_data.image.path, share, png_jpeg_only=True)
        params["MultiViewImages"] = [
            {"ViewType": v.view, "ViewImageBase64": encode_image(v.image.path, share, png_jpeg_only=True)} for v in input_data.views
        ]
        return params

    @function(annotations={"title": "Generate 3D model", "readOnlyHint": False, "destructiveHint": False, "openWorldHint": True})
    async def run(self, input_data: AppInput) -> AppOutput:
        """Make a 3D model from a text prompt, an image, or a front image with side views."""
        mode = "text" if input_data.image is None else ("multiview" if input_data.views else "image")
        credits = credits_for(input_data)
        self.logger.info(
            "%s to 3D: %s, pbr %s, faces %s, views %s, region %s",
            mode, input_data.generate_type, input_data.enable_pbr, input_data.face_count or "default",
            ",".join(v.view for v in input_data.views) or "-", self.tencent.region,
        )
        job = await self.tencent.run_job("SubmitHunyuanTo3DProJob", "QueryHunyuanTo3DProJob", self._params(input_data), "Hunyuan 3D")
        model, preview, others = await result_files(job, "Hunyuan 3D")

        self.logger.info("usage: credits=%d mode=%s type=%s model=%s", credits, mode, input_data.generate_type, MODEL)
        meta = RawMeta(
            extra={
                "credits": credits,
                "mode": mode,
                "model": MODEL,
                "generate_type": input_data.generate_type,
                "enable_pbr": input_data.enable_pbr,
                "multiview": bool(input_data.views),
                "face_count_set": input_data.face_count is not None,
            }
        )
        return AppOutput(
            model=model,
            preview=preview,
            files=others,
            output_meta=OutputMeta(inputs=[], outputs=[meta]),
        )

    async def unload(self):
        await self.tencent.close()

import logging
import os
import tempfile
from typing import List, Literal, Optional

import httpx
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from PIL import Image
from pydantic import Field

from .minimax_image import (
    MiniMaxImageError,
    build_request,
    download,
    encode_references,
    post_images,
    usage_extra,
)

MODEL = "MiniMax-H3-Image"
EXTENSIONS = {"jpeg": "jpg", "png": "png", "webp": "webp"}


class AppInput(BaseAppInput):
    """MiniMax H3 Image. Text-to-image from a prompt, or an edit when reference images are given."""

    prompt: str = Field(
        min_length=1,
        max_length=32000,
        pattern=r"\S",
        description="What to generate, or how to change the reference images. Refer to references as Picture 1, Picture 2, in upload order. Up to 32000 characters.",
        examples=["A ceramic teapot shaped like a lighthouse on a weathered wooden table, soft morning window light, shallow depth of field, product photography."],
    )
    images: List[File] = Field(
        default_factory=list,
        max_length=16,
        description="Reference images for editing or composition, up to 16. JPG, PNG, WEBP, HEIC or HEIF; each side 128 to 8192 px; long side at most 4x the short side. Leave empty for text-to-image.",
    )
    aspect_ratio: str = Field(
        default="auto",
        pattern=r"^(auto|[1-9][0-9]{0,2}:[1-9][0-9]{0,2})$",
        description="Output aspect ratio as X:Y (e.g. 16:9, 3:4, 21:9), between 1:3 and 3:1, or auto to let the model choose from the prompt and references.",
        examples=["16:9"],
    )
    resolution: Literal["1K", "2K", "4K"] = Field(
        default="2K",
        description="Output area: 1K (about 1024x1024), 2K (about 2048x2048) or 4K (about 4096x4096). Width and height follow the aspect ratio, e.g. 2K at 16:9 is 2720x1536.",
    )
    size: Optional[str] = Field(
        default=None,
        pattern=r"^[1-9][0-9]{2,3}x[1-9][0-9]{2,3}$",
        description="Exact output size as WIDTHxHEIGHT, e.g. 2720x1536. Overrides aspect_ratio and resolution. Area 655,360 to 20,000,000 px, ratio between 1:3 and 3:1.",
    )
    num_images: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Number of images to generate.",
    )
    enhance_prompt: bool = Field(
        default=True,
        description="Let the model expand the prompt before generating. Always on when reference images are given. When off with aspect_ratio auto, the image is 1:1.",
    )
    output_format: Literal["jpeg", "png", "webp"] = Field(
        default="jpeg",
        description="Image file format.",
    )


class AppOutput(BaseAppOutput):
    images: List[File] = Field(description="The generated images.")
    revised_prompts: List[str] = Field(
        default_factory=list,
        description="The prompt the model used for each image, after enhancement.",
    )


def _get_api_key() -> str:
    key = os.environ.get("MINIMAX_KEY")
    if not key:
        raise RuntimeError("MINIMAX_KEY is not set. Connect the MiniMax credential for this workspace.")
    return key.strip()


class App(BaseApp):

    async def setup(self, config):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        self.api_key = _get_api_key()
        self.client = httpx.AsyncClient()
        self.logger.info(f"{MODEL} initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        try:
            image_uris, input_dims = encode_references(input_data.images)
        except ValueError as exc:
            raise ValueError(f"Reference image refused: {exc}") from exc

        path, body = build_request(
            MODEL,
            input_data.prompt,
            image_uris,
            n=input_data.num_images,
            size=input_data.size,
            aspect_ratio=input_data.aspect_ratio,
            area=input_data.resolution,
            enhance_prompt=input_data.enhance_prompt,
            output_format=input_data.output_format,
        )
        mode = "image-edit" if image_uris else "text-to-image"
        summary = {k: v for k, v in body.items() if k not in ("prompt", "images")}
        self.logger.info(
            f"Mode: {mode}, endpoint: {path}, references: {len(image_uris)}, "
            f"prompt chars: {len(body['prompt'])}, params: {summary}"
        )

        result = await post_images(self.client, self.api_key, path, body, self.logger)
        usage = result.get("usage") or {}
        self.logger.info(f"Request {result.get('request_id')}: usage {usage}")

        out_dir = tempfile.mkdtemp(prefix="h3-image-")
        ext = EXTENSIONS.get(result.get("output_format") or input_data.output_format, "jpg")
        images: List[File] = []
        revised: List[str] = []
        failures: List[str] = []
        widths: List[int] = []
        heights: List[int] = []
        for index, item in enumerate(result.get("data") or []):
            if item.get("error") or not item.get("url"):
                err = item.get("error") or {}
                failures.append(f"image {index + 1}: {err.get('message', 'no image returned')} ({err.get('code', '?')})")
                continue
            file_path = await download(self.client, item["url"], os.path.join(out_dir, f"image-{index + 1}.{ext}"))
            with Image.open(file_path) as img:
                width, height = img.size
            if item.get("size") and item["size"] != f"{width}x{height}":
                self.logger.warning(f"Image {index + 1}: reported size {item['size']}, file is {width}x{height}")
            widths.append(width)
            heights.append(height)
            images.append(File(path=file_path))
            revised.append(item.get("revised_prompt") or "")

        for failure in failures:
            self.logger.warning(f"Not generated, not billed: {failure}")
        if not images:
            raise MiniMaxImageError(200, "no image was generated: " + "; ".join(failures or ["empty response"]))

        extra = usage_extra(usage)
        extra.update({
            "model": MODEL,
            "mode": mode,
            "resolution": "custom" if input_data.size else input_data.resolution,
            "requested_images": input_data.num_images,
            "input_image_count": len(image_uris),
            "total_pixels": sum(w * h for w, h in zip(widths, heights)),
            "max_pixels": max(w * h for w, h in zip(widths, heights)),
        })
        output_meta = OutputMeta(
            inputs=[ImageMeta(width=w, height=h, count=1) if w and h else ImageMeta(count=1) for w, h in input_dims],
            outputs=[ImageMeta(width=widths[0], height=heights[0], count=len(images), extra=extra)],
        )

        self.logger.info(
            f"Generated {len(images)}/{input_data.num_images} image(s), first {widths[0]}x{heights[0]}; "
            f"tokens in text {extra['input_text_tokens']}, in image {extra['input_image_tokens']}, "
            f"out image {extra['output_image_tokens']}"
        )
        return AppOutput(images=images, revised_prompts=revised, output_meta=output_meta)

    async def on_cancel(self):
        self.logger.info("Cancellation requested")
        return True

    async def unload(self):
        await self.client.aclose()

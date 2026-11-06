"""Shared input, request building and output handling for the GPT Image 2.6 apps."""

import logging
from typing import Literal, Optional

from inferencesh import BaseAppInput, BaseAppOutput, File, ImageMeta, OutputMeta
from pydantic import Field

from .openai_helper import make_size_string, moderated, save_base64_image, validate_and_fix_dimensions

ComplexityType = Literal["low", "medium", "high"]
QualityType = Literal["auto", "low", "medium", "high", "xhigh", "max"]
FormatType = Literal["png", "jpeg", "webp", "gif", "svg"]
ModerationLevel = Literal["auto", "low"]
BackgroundType = Literal["auto", "transparent", "opaque"]

# svg output needs an author model next to the image model; its tokens are billed separately.
SVG_AUTHOR_MODEL = "gpt-6-luna"


class ImageInput(BaseAppInput):
    prompt: str = Field(
        description="Text prompt describing the desired image. For gif output, describe the subject, motion and loop.",
        max_length=32000,
        examples=["A cat wearing a tiny top hat, oil painting style"],
    )
    images: Optional[list[File]] = Field(
        default=None,
        description="Optional image(s) to edit. A mask or markup applies to the first image; "
        "the others serve as references.",
    )
    mask: Optional[File] = Field(
        default=None,
        description="Optional mask for the first image. Its transparent areas mark where to edit. "
        "Cannot be combined with markup.",
    )
    markup: Optional[File] = Field(
        default=None,
        description="Optional transparent PNG with circles, sketches or other annotations showing where to edit. "
        "Must match the first image's dimensions. Cannot be combined with mask.",
    )
    width: int = Field(
        default=1024,
        ge=256,
        le=3840,
        description="Output width in pixels. Must be a multiple of 16. For gif this is the frame width; "
        "leave width and height unset to let the model pick a size that fits the frame count.",
    )
    height: int = Field(
        default=1024,
        ge=256,
        le=3840,
        description="Output height in pixels. Must be a multiple of 16. For gif this is the frame height; "
        "leave width and height unset to let the model pick a size that fits the frame count.",
    )
    quality: QualityType = Field(
        default="auto",
        description="Rendering quality. 'low' for drafts, 'high' for final assets, "
        "'xhigh' and 'max' for maximum detail at higher cost. For svg, sets the reference image's quality.",
    )
    complexity: ComplexityType = Field(
        default="high",
        description="How much the model plans before rendering, independent of quality and at no extra cost. "
        "Lower suits focused edits; higher helps dense text and complex layouts but adds latency.",
    )
    n: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Number of images to generate (1-10). svg output requires 1.",
    )
    output_format: FormatType = Field(
        default="png",
        description="Output file format. gif (animated) and svg (vector) are for generation only, not edits.",
    )
    output_compression: Optional[int] = Field(
        default=None,
        ge=0,
        le=100,
        description="Compression level for jpeg/webp (0-100). Ignored for other formats.",
    )
    frame_count: int = Field(
        default=9,
        ge=2,
        le=16,
        description="Number of animation frames. Used only for gif output.",
    )
    frame_duration_ms: int = Field(
        default=200,
        ge=20,
        le=2000,
        description="Delay per animation frame in milliseconds, rounded to 10 ms. Used only for gif output.",
    )
    moderation: ModerationLevel = Field(
        default="auto",
        description="Content moderation strictness. 'auto' applies standard filtering; 'low' is less restrictive.",
    )
    background: BackgroundType = Field(
        default="auto",
        description="Background transparency. 'transparent' requires png or webp output; "
        "prompt for an isolated subject. 'opaque' forces a solid background.",
    )


class ImageOutput(BaseAppOutput):
    images: list[File] = Field(description="The generated image files.")


def _validate(input_data: ImageInput, is_edit: bool) -> None:
    fmt = input_data.output_format
    if input_data.background == "transparent" and fmt not in ("png", "webp"):
        raise ValueError("Transparent backgrounds require png or webp output_format")
    if input_data.mask and input_data.markup:
        raise ValueError("mask and markup cannot be combined")
    if (input_data.mask or input_data.markup) and not is_edit:
        raise ValueError("mask and markup require input images")
    if fmt in ("gif", "svg") and is_edit:
        raise ValueError(f"{fmt} output is for generation only; image edits do not support it")
    if fmt == "svg" and input_data.n != 1:
        raise ValueError("svg output requires n=1")


def _common_kwargs(model: str, input_data: ImageInput, size: str) -> dict:
    kwargs: dict = {
        "model": model,
        "prompt": input_data.prompt,
        "n": input_data.n,
        "size": size,
        "quality": input_data.quality,
        "complexity": input_data.complexity,
        "output_format": input_data.output_format,
        "background": input_data.background,
    }
    if input_data.output_compression is not None and input_data.output_format in ("jpeg", "webp"):
        kwargs["output_compression"] = input_data.output_compression
    if input_data.output_format == "gif":
        kwargs["frame_count"] = input_data.frame_count
        kwargs["frame_duration_ms"] = input_data.frame_duration_ms
    if input_data.output_format == "svg":
        kwargs["svg"] = {"model": SVG_AUTHOR_MODEL}
    return kwargs


async def _generate(client, model: str, input_data: ImageInput, size: str):
    return await client.images.generate(
        **_common_kwargs(model, input_data, size),
        moderation=input_data.moderation,
    )


async def _edit(client, model: str, input_data: ImageInput, size: str):
    open_files = []

    def _open(path: str):
        handle = open(path, "rb")
        open_files.append(handle)
        return handle

    try:
        image_files = [_open(img.path) for img in input_data.images if img is not None]
        if not image_files:
            raise ValueError("No valid images provided for editing")
        kwargs = _common_kwargs(model, input_data, size)
        kwargs["image"] = image_files if len(image_files) > 1 else image_files[0]
        if input_data.mask and input_data.mask.exists():
            kwargs["mask"] = _open(input_data.mask.path)
        if input_data.markup and input_data.markup.exists():
            kwargs["markup"] = _open(input_data.markup.path)
        return await client.images.edit(**kwargs, extra_body={"moderation": input_data.moderation})
    finally:
        for handle in open_files:
            handle.close()


def _usage_extra(response, logger: logging.Logger) -> dict:
    extra: dict = {}
    usage = getattr(response, "usage", None)
    if usage is not None:
        extra["input_tokens"] = getattr(usage, "input_tokens", None)
        extra["output_tokens"] = getattr(usage, "output_tokens", None)
        details = getattr(usage, "input_tokens_details", None)
        if details is not None:
            extra["input_text_tokens"] = getattr(details, "text_tokens", None)
            extra["input_image_tokens"] = getattr(details, "image_tokens", None)
    svg_usage = getattr(response, "svg_usage", None)
    if svg_usage is not None:
        extra["svg_model"] = svg_usage.model
        extra["svg_input_tokens"] = svg_usage.input_tokens
        extra["svg_output_tokens"] = svg_usage.output_tokens
        extra["svg_reasoning_tokens"] = svg_usage.reasoning_tokens
    logger.info(f"Usage: {extra}")
    return extra


async def run_image(client, model: str, input_data: ImageInput, logger: logging.Logger):
    """Generate or edit images; returns (files, output_meta)."""
    is_edit = bool(input_data.images)
    _validate(input_data, is_edit)

    fmt = input_data.output_format
    width, height = input_data.width, input_data.height
    if fmt != "gif":
        width, height = validate_and_fix_dimensions(width, height, logger=logger)
    size = make_size_string(width, height)
    # A gif's frames share one sheet, so the default 1024x1024 fits no frame count: let OpenAI pick.
    if fmt == "gif" and not {"width", "height"} & input_data.model_fields_set:
        size = "auto"
    mode = "edit" if is_edit else "generate"

    logger.info(f"Starting {mode} with {model}: {input_data.prompt[:100]}")
    logger.info(
        f"size={size} quality={input_data.quality} complexity={input_data.complexity} "
        f"n={input_data.n} format={fmt} background={input_data.background}"
    )

    if is_edit:
        response = await moderated(_edit(client, model, input_data, size))
    else:
        response = await moderated(_generate(client, model, input_data, size))

    files = [File(path=save_base64_image(img.b64_json, fmt)) for img in response.data or []]
    if not files:
        raise RuntimeError("No images generated")

    out_w, out_h = width, height
    if fmt != "svg":
        try:
            from PIL import Image as PILImage

            with PILImage.open(files[0].path) as img:
                out_w, out_h = img.size
        except Exception as exc:
            logger.warning(f"Could not read output dimensions: {exc}")

    extra = {
        "mode": mode,
        "model": model,
        "quality": getattr(response, "quality", None) or input_data.quality,
        "complexity": input_data.complexity,
        "background": input_data.background,
        "output_format": fmt,
        **_usage_extra(response, logger),
    }
    if fmt == "gif":
        extra["frame_count"] = input_data.frame_count

    logger.info(f"Generated {len(files)} image(s) at {out_w}x{out_h}")
    meta = OutputMeta(
        outputs=[ImageMeta(width=out_w, height=out_h, count=len(files), extra=extra)]
    )
    return files, meta

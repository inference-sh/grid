"""
Seedream 5.5 - BytePlus ModelArk image generation

ByteDance's Seedream 5.5 (model id dola-seedream-5-5-261028) for text-to-image,
image editing and multi-reference generation (up to 10 reference images), with
native transparent-background (RGBA) output.
Uses the BytePlus ARK SDK with synchronous image generation.

Pricing (BytePlus model pricing page):
  input image: $0.008 each, from the first
  output: $0.020 for 921,600 <= P <= 2,601,123.84 pixels (1K, 1.5K)
          $0.035 for P <= 4,624,220.16 pixels (2K)
See pricing.md.
"""

import base64
import io
import logging
from enum import Enum
from typing import List, Optional, Tuple

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from inferencesh.images import ImageLimits, fit_image
from pydantic import Field, model_validator

from .byteplus_helper import setup_byteplus_client
from .download_helper import download_file

MODEL_ID = "dola-seedream-5-5-261028"
MAX_REFERENCE_IMAGES = 10

# Reference images are sent inline as data URLs. Upstream accepts up to 30 MB and
# 36,000,000 pixels per image; we downsize (never upscale) to 10 MB to keep a
# 10-image request body reasonable.
INPUT_LIMITS = ImageLimits(
    max_pixels=36_000_000,
    max_bytes=10 * 1024 * 1024,
    mime_types=("image/png", "image/jpeg", "image/webp", "image/bmp", "image/tiff", "image/gif"),
)

# Output pixel tier boundary from the pricing page: 1536^2 * 1.05^2.
TIER_LOW_MAX_PIXELS = 2_601_123.84


class SizeEnum(str, Enum):
    """Output resolution tier."""
    size_1k = "1K"
    size_1_5k = "1.5K"
    size_2k = "2K"


class AspectRatioEnum(str, Enum):
    """Output aspect ratio."""
    ratio_1_1 = "1:1"
    ratio_4_3 = "4:3"
    ratio_3_4 = "3:4"
    ratio_16_9 = "16:9"
    ratio_9_16 = "9:16"
    ratio_3_2 = "3:2"
    ratio_2_3 = "2:3"
    ratio_21_9 = "21:9"


class OutputFormatEnum(str, Enum):
    """Output image format."""
    png = "png"
    jpeg = "jpeg"


class BackgroundEnum(str, Enum):
    """Transparency mode of the output image."""
    auto = "auto"
    transparent = "transparent"
    opaque = "opaque"


# (size, aspect_ratio) -> (width, height), from the official Seedream 5.5 size table.
# Total pixels must stay in [921,600, 4,624,220] upstream.
DIMENSIONS = {
    ("1K", "1:1"): (1024, 1024),
    ("1K", "4:3"): (1152, 864),
    ("1K", "3:4"): (864, 1152),
    ("1K", "16:9"): (1424, 800),
    ("1K", "9:16"): (800, 1424),
    ("1K", "3:2"): (1248, 832),
    ("1K", "2:3"): (832, 1248),
    ("1K", "21:9"): (1568, 672),
    ("1.5K", "1:1"): (1536, 1536),
    ("1.5K", "4:3"): (1792, 1344),
    ("1.5K", "3:4"): (1344, 1792),
    ("1.5K", "16:9"): (2048, 1152),
    ("1.5K", "9:16"): (1152, 2048),
    ("1.5K", "3:2"): (1872, 1248),
    ("1.5K", "2:3"): (1248, 1872),
    ("1.5K", "21:9"): (2352, 1008),
    ("2K", "1:1"): (2048, 2048),
    ("2K", "4:3"): (2368, 1776),
    ("2K", "3:4"): (1776, 2368),
    ("2K", "16:9"): (2816, 1584),
    ("2K", "9:16"): (1584, 2816),
    ("2K", "3:2"): (2496, 1664),
    ("2K", "2:3"): (1664, 2496),
    ("2K", "21:9"): (3136, 1344),
}


class AppInput(BaseAppInput):
    """Input schema for Seedream 5.5 image generation."""

    prompt: str = Field(
        min_length=1,
        description="Text prompt describing the image to generate or the edit to perform. Best under about 600 English words. Supports text rendering in 16 languages.",
        examples=["A glass perfume bottle on a pale stone pedestal, soft morning light, product photography"],
    )
    images: List[File] = Field(
        default_factory=list,
        max_length=MAX_REFERENCE_IMAGES,
        description="Optional reference images for editing or multi-reference generation, up to 10. jpeg, png, webp, bmp, tiff or gif; aspect ratio between 1:16 and 16:1.",
    )
    size: SizeEnum = Field(
        default=SizeEnum.size_2k,
        description="Output resolution tier. 1K and 1.5K are billed at the lower rate, 2K at the higher rate.",
    )
    aspect_ratio: AspectRatioEnum = Field(
        default=AspectRatioEnum.ratio_1_1,
        description="Output aspect ratio.",
    )
    output_format: OutputFormatEnum = Field(
        default=OutputFormatEnum.png,
        description="Output image format. Transparent backgrounds need png.",
    )
    background: BackgroundEnum = Field(
        default=BackgroundEnum.auto,
        description="Transparency of the output. auto lets the model decide from the prompt, transparent returns an RGBA png, opaque returns an RGB image.",
    )
    watermark: bool = Field(
        default=False,
        description="Add an 'AI generated' watermark to the lower-right corner.",
    )

    @model_validator(mode="after")
    def _check_background_format(self):
        if self.background == BackgroundEnum.transparent and self.output_format == OutputFormatEnum.jpeg:
            raise ValueError("background 'transparent' requires output_format 'png' (jpeg has no alpha channel).")
        return self


class AppOutput(BaseAppOutput):
    """Output schema for Seedream 5.5 image generation."""

    image: File = Field(description="The generated image file.")


def _parse_size(value: Optional[str]) -> Optional[Tuple[int, int]]:
    """Parse an upstream '<w>x<h>' string."""
    if not value or "x" not in value:
        return None
    try:
        w, h = value.lower().split("x", 1)
        return int(w), int(h)
    except ValueError:
        return None


class App(BaseApp):
    """Seedream 5.5 image generation via the BytePlus ARK SDK."""

    async def setup(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        logging.getLogger("httpx").setLevel(logging.WARNING)

        self.client = setup_byteplus_client()
        self.model_id = MODEL_ID
        self.logger.info(f"Seedream 5.5 initialized with model: {self.model_id}")

    def _image_param(self, img: File, index: int) -> Tuple[str, ImageMeta]:
        """One reference image as a data URL (local copy) or its URL, with its metadata."""
        if img is None:
            raise RuntimeError(f"Reference image {index + 1} is empty.")
        path = getattr(img, "path", None)
        uri = getattr(img, "uri", None)
        if path and img.exists():
            try:
                data, mime = fit_image(path, INPUT_LIMITS)
            except Exception as e:
                if uri and uri.startswith(("http://", "https://")):
                    self.logger.warning(f"Could not fit reference image {index + 1} ({e}); sending its URL")
                    return uri, ImageMeta()
                raise RuntimeError(f"Reference image {index + 1} could not be read: {e}")
            width = height = None
            try:
                from PIL import Image
                with Image.open(io.BytesIO(data)) as im:
                    width, height = im.size
            except Exception:
                pass
            if width and height:
                if width <= 14 or height <= 14:
                    raise RuntimeError(f"Reference image {index + 1} is {width}x{height}; each side must be over 14 px.")
                ratio = width / height
                if ratio < 1 / 16 or ratio > 16:
                    raise RuntimeError(f"Reference image {index + 1} aspect ratio must be between 1:16 and 16:1.")
            url = "data:" + mime.lower() + ";base64," + base64.b64encode(data).decode("ascii")
            return url, ImageMeta(width=width, height=height)
        if uri and uri.startswith(("http://", "https://")):
            return uri, ImageMeta()
        raise RuntimeError(f"Reference image {index + 1} has no readable file or URL.")

    async def run(self, input_data: AppInput) -> AppOutput:
        images = [img for img in (input_data.images or []) if img is not None]
        if len(images) > 1:
            mode = "multi-image-to-image"
        elif len(images) == 1:
            mode = "image-to-image"
        else:
            mode = "text-to-image"

        req_w, req_h = DIMENSIONS[(input_data.size.value, input_data.aspect_ratio.value)]
        size_str = f"{req_w}x{req_h}"
        fmt = input_data.output_format.value
        background = input_data.background.value

        self.logger.info(f"Starting {mode}: size={size_str} format={fmt} background={background} images={len(images)}")
        self.logger.info(f"Prompt: {input_data.prompt[:100]}")

        image_param = None
        input_metas: List[ImageMeta] = []
        if images:
            urls = []
            for i, img in enumerate(images):
                url, meta = self._image_param(img, i)
                urls.append(url)
                input_metas.append(meta)
            image_param = urls[0] if len(urls) == 1 else urls

        # Seedream 5.5 has no sequential generation, streaming, guidance_scale or
        # optimize_prompt fast mode. The SDK has no `background` kwarg, so it goes in extra_body.
        try:
            result = self.client.images.generate(
                model=self.model_id,
                prompt=input_data.prompt,
                image=image_param,
                size=size_str,
                output_format=fmt,
                response_format="url",
                watermark=input_data.watermark,
                extra_body={"background": background},
            )
        except Exception as e:
            self.logger.error(f"Seedream 5.5 request failed: {e}")
            raise RuntimeError(f"Image generation failed: {e}")

        usage = getattr(result, "usage", None)
        if usage is not None:
            self.logger.info(
                f"Usage: generated_images={getattr(usage, 'generated_images', None)} "
                f"output_tokens={getattr(usage, 'output_tokens', None)}"
            )

        data = getattr(result, "data", None) or []
        item = data[0] if data else None
        if item is None:
            raise RuntimeError("Image generation failed: no image in the response")
        item_error = getattr(item, "error", None)
        if item_error:
            raise RuntimeError(f"Image generation failed: {item_error}")
        image_url = getattr(item, "url", None)
        if not image_url:
            self.logger.error(f"No image URL in response: {result}")
            raise RuntimeError("Image generation failed: no image URL in the response")

        # URLs expire within 24 h: download now.
        image_path = download_file(image_url, suffix=".png" if fmt == "png" else ".jpg", logger=self.logger)

        out_w, out_h = _parse_size(getattr(item, "size", None)) or (req_w, req_h)
        pixels = out_w * out_h
        tier = "low" if pixels <= TIER_LOW_MAX_PIXELS else "high"
        actual_background = getattr(item, "background", None)
        self.logger.info(f"Generated {out_w}x{out_h} ({tier} tier), background={actual_background}")

        output_meta = OutputMeta(
            inputs=input_metas,
            outputs=[
                ImageMeta(
                    width=out_w,
                    height=out_h,
                    count=1,
                    extra={
                        "mode": mode,
                        "size": input_data.size.value,
                        "pixel_tier": tier,
                        "input_images": len(images),
                        "format": fmt,
                        "background": background,
                        "actual_background": actual_background,
                        "watermark": input_data.watermark,
                    },
                )
            ],
        )

        return AppOutput(image=File(path=image_path), output_meta=output_meta)

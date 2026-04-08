"""
Image Resize - Resize images by dimensions, scale factor, or megapixel target
"""

import math
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field
from typing import Optional, Literal
from PIL import Image
import logging


class AppInput(BaseAppInput):
    """Input schema for Image Resize."""

    image: File = Field(description="Input image to resize.")
    width: Optional[int] = Field(
        default=None, ge=1, le=16384,
        description="Target width in pixels. If only width is set, height is calculated to preserve aspect ratio."
    )
    height: Optional[int] = Field(
        default=None, ge=1, le=16384,
        description="Target height in pixels. If only height is set, width is calculated to preserve aspect ratio."
    )
    scale: Optional[float] = Field(
        default=None, gt=0, le=10,
        description="Scale factor (e.g. 0.5 = half size, 2.0 = double). Overridden by width/height if set."
    )
    megapixels: Optional[float] = Field(
        default=None, gt=0, le=100,
        description="Target resolution in megapixels, preserving aspect ratio. Overridden by width/height/scale."
    )
    resample: Literal["lanczos", "bicubic", "bilinear", "nearest"] = Field(
        default="lanczos",
        description="Resampling filter."
    )
    output_format: Literal["jpg", "png", "webp"] = Field(
        default="jpg",
        description="Output image format."
    )
    output_quality: int = Field(
        default=85, ge=0, le=100,
        description="Output quality (0-100). Not relevant for PNG."
    )


class AppOutput(BaseAppOutput):
    """Output schema for Image Resize."""

    image: File = Field(description="Resized image.")
    width: int = Field(description="Output width in pixels.")
    height: int = Field(description="Output height in pixels.")


RESAMPLE_MAP = {
    "lanczos": Image.LANCZOS,
    "bicubic": Image.BICUBIC,
    "bilinear": Image.BILINEAR,
    "nearest": Image.NEAREST,
}


class App(BaseApp):
    """Image Resize utility."""

    async def setup(self, metadata):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info("Image Resize initialized")

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Resize an image."""
        if not input_data.image.exists():
            raise RuntimeError(f"Image does not exist: {input_data.image.path}")

        img = Image.open(input_data.image.path)
        orig_w, orig_h = img.size
        self.logger.info(f"Input: {orig_w}x{orig_h}")

        # Determine target size
        if input_data.width or input_data.height:
            if input_data.width and input_data.height:
                new_w, new_h = input_data.width, input_data.height
            elif input_data.width:
                ratio = input_data.width / orig_w
                new_w = input_data.width
                new_h = round(orig_h * ratio)
            else:
                ratio = input_data.height / orig_h
                new_w = round(orig_w * ratio)
                new_h = input_data.height
        elif input_data.scale:
            new_w = round(orig_w * input_data.scale)
            new_h = round(orig_h * input_data.scale)
        elif input_data.megapixels:
            target_pixels = input_data.megapixels * 1_000_000
            current_pixels = orig_w * orig_h
            ratio = math.sqrt(target_pixels / current_pixels)
            new_w = round(orig_w * ratio)
            new_h = round(orig_h * ratio)
        else:
            raise RuntimeError("Specify at least one of: width, height, scale, or megapixels")

        new_w = max(1, new_w)
        new_h = max(1, new_h)

        self.logger.info(f"Resizing to {new_w}x{new_h}")

        resample = RESAMPLE_MAP[input_data.resample]
        resized = img.resize((new_w, new_h), resample)

        # Handle RGB conversion for JPEG
        fmt = input_data.output_format
        if fmt == "jpg" and resized.mode in ("RGBA", "P"):
            resized = resized.convert("RGB")

        output_path = f"/tmp/resized.{fmt}"
        if fmt == "jpg":
            resized.save(output_path, "JPEG", quality=input_data.output_quality)
        elif fmt == "webp":
            resized.save(output_path, "WEBP", quality=input_data.output_quality)
        else:
            resized.save(output_path, "PNG")

        self.logger.info(f"Saved {fmt} to {output_path}")

        return AppOutput(
            image=File(path=output_path),
            width=new_w,
            height=new_h,
            output_meta=OutputMeta(
                outputs=[ImageMeta(width=new_w, height=new_h, count=1)]
            ),
        )

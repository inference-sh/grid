"""
Seedream 5.0 Pro - BytePlus Image Generation

ByteDance's flagship image model for precision creation and editing.
Supports text-to-image, single image editing, and multi-reference (up to 10 images)
generation with pixel-level interactive editing, intelligent layer understanding,
and native multilingual text rendering (14 languages).
Uses BytePlus ARK SDK with synchronous image generation.

Pricing: $0.045 per image (<= 2.36MP), $0.09 per image (> 2.36MP).
Input images: first free, $0.003 each additional.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field
from typing import Optional, List, Tuple
from enum import Enum
import logging

from .byteplus_helper import (
    setup_byteplus_client,
    download_image,
)


class SizeEnum(str, Enum):
    """Image size/resolution options."""
    size_1k = "1K"
    size_2k = "2K"


class AspectRatioEnum(str, Enum):
    """Aspect ratio options."""
    ratio_1_1 = "1:1"
    ratio_3_4 = "3:4"
    ratio_4_3 = "4:3"
    ratio_16_9 = "16:9"
    ratio_9_16 = "9:16"
    ratio_3_2 = "3:2"
    ratio_2_3 = "2:3"
    ratio_21_9 = "21:9"


# Dimension lookup: (size, aspect_ratio) -> (width, height)
# Constraints: total pixels in [1280x720=921,600, 2048x2048=4,194,304],
# width/height multiples of 16.
DIMENSIONS = {
    # 1K dimensions
    ("1K", "1:1"): (1024, 1024),
    ("1K", "3:4"): (864, 1152),
    ("1K", "4:3"): (1152, 864),
    ("1K", "16:9"): (1280, 720),
    ("1K", "9:16"): (720, 1280),
    ("1K", "3:2"): (1248, 832),
    ("1K", "2:3"): (832, 1248),
    ("1K", "21:9"): (1568, 672),
    # 2K dimensions
    ("2K", "1:1"): (2048, 2048),
    ("2K", "3:4"): (1728, 2304),
    ("2K", "4:3"): (2304, 1728),
    ("2K", "16:9"): (2560, 1440),
    ("2K", "9:16"): (1440, 2560),
    ("2K", "3:2"): (2496, 1664),
    ("2K", "2:3"): (1664, 2496),
    ("2K", "21:9"): (2944, 1264),
}


class OutputFormatEnum(str, Enum):
    """Output image format."""
    png = "png"
    jpeg = "jpeg"


class AppInput(BaseAppInput):
    """Input schema for Seedream 5.0 Pro image generation."""

    prompt: str = Field(
        description="Text prompt describing the image to generate or the edit to perform. Supports complex instructions: infographics, precise regional edits, layer-aware design, and text rendering in 14 languages.",
        examples=["A detailed infographic explaining the water cycle, clean editorial design, labeled arrows, soft color palette"]
    )
    images: Optional[List[File]] = Field(
        default=None,
        description="Optional reference images for image editing or multi-reference generation. Up to 10 images supported."
    )
    size: SizeEnum = Field(
        default=SizeEnum.size_2k,
        description="Output image resolution. 1K or 2K."
    )
    aspect_ratio: AspectRatioEnum = Field(
        default=AspectRatioEnum.ratio_1_1,
        description="Output image aspect ratio."
    )
    output_format: OutputFormatEnum = Field(
        default=OutputFormatEnum.png,
        description="Output image format (png or jpeg)."
    )
    watermark: bool = Field(
        default=False,
        description="Whether to add a watermark to the generated image."
    )


class AppOutput(BaseAppOutput):
    """Output schema for Seedream 5.0 Pro image generation."""

    image: File = Field(description="The generated image file.")


class App(BaseApp):
    """Seedream 5.0 Pro image generation application using BytePlus ARK SDK."""

    async def setup(self):
        """Initialize the BytePlus client."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Suppress noisy httpx logs
        logging.getLogger("httpx").setLevel(logging.WARNING)

        # Initialize client
        self.client = setup_byteplus_client()
        self.model_id = "dola-seedream-5-0-pro-260628"

        self.logger.info(f"Seedream 5.0 Pro initialized with model: {self.model_id}")

    def get_dimensions(self, size: SizeEnum, aspect_ratio: AspectRatioEnum) -> Tuple[int, int]:
        """Get pixel dimensions for size and aspect ratio combination."""
        key = (size.value, aspect_ratio.value)
        return DIMENSIONS.get(key, (2048, 2048))

    async def run(self, input_data: AppInput) -> AppOutput:
        """Generate image using Seedream 5.0 Pro."""
        try:
            # Determine mode based on input
            if input_data.images and len(input_data.images) > 1:
                mode = "multi-image-to-image"
            elif input_data.images and len(input_data.images) == 1:
                mode = "image-to-image"
            else:
                mode = "text-to-image"

            # Get dimensions for size/aspect_ratio combination
            width, height = self.get_dimensions(input_data.size, input_data.aspect_ratio)
            size_str = f"{width}x{height}"

            self.logger.info(f"Starting {mode} generation")
            self.logger.info(f"Prompt: {input_data.prompt[:100]}...")
            self.logger.info(f"Size: {size_str}, Format: {input_data.output_format.value}")

            # Build image parameter (single URL, list of URLs, or None)
            image_param = None
            input_image_count = 0
            if input_data.images:
                if len(input_data.images) > 10:
                    raise RuntimeError("Seedream 5.0 Pro supports up to 10 reference images.")
                image_urls = []
                for img in input_data.images:
                    if not img.exists():
                        raise RuntimeError(f"Input image does not exist at path: {img.path}")
                    image_urls.append(img.uri)
                input_image_count = len(image_urls)

                # Single image: pass as string; multiple: pass as list
                image_param = image_urls[0] if len(image_urls) == 1 else image_urls

            # Call image generation API with WIDTHxHEIGHT format.
            # Note: seedream-5-0-pro does NOT support sequential_image_generation,
            # stream, or guidance_scale — passing them causes an API error.
            result = self.client.images.generate(
                model=self.model_id,
                prompt=input_data.prompt,
                size=size_str,
                output_format=input_data.output_format.value,
                response_format="url",
                watermark=input_data.watermark,
                image=image_param,
            )

            # Extract image URL from response
            if not result.data or len(result.data) == 0:
                raise RuntimeError("No image data in response")

            image_url = result.data[0].url
            if not image_url:
                self.logger.error(f"Could not extract image URL from result: {result}")
                raise RuntimeError("Failed to get image URL from response")

            # Download image
            image_path = download_image(image_url, self.logger)

            # Build input metadata for pricing
            input_metas = []
            if input_data.images:
                for _ in input_data.images:
                    input_metas.append(ImageMeta())

            # Build output metadata for pricing
            output_meta = OutputMeta(
                inputs=input_metas,
                outputs=[
                    ImageMeta(
                        width=width,
                        height=height,
                        extra={
                            "mode": mode,
                            "input_images": input_image_count,
                            "watermark": input_data.watermark,
                            "format": input_data.output_format.value,
                        }
                    )
                ]
            )

            self.logger.info(f"Image generated successfully: {image_path}")

            return AppOutput(
                image=File(path=image_path),
                output_meta=output_meta,
            )

        except Exception as e:
            self.logger.error(f"Error during image generation: {e}")
            raise RuntimeError(f"Image generation failed: {str(e)}")

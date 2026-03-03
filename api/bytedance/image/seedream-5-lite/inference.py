"""
Seedream 5 Lite - BytePlus Image Generation

Generate high-quality images from text prompts with single or multi-image input.
Supports text-to-image, image-to-image, and multi-reference image blending.
Uses BytePlus ARK SDK with synchronous image generation.

Pricing: $0.035 per image (text-to-image and image-to-image)
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
    size_2k = "2K"
    size_3k = "3K"


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
DIMENSIONS = {
    # 2K dimensions
    ("2K", "1:1"): (2048, 2048),
    ("2K", "3:4"): (1728, 2304),
    ("2K", "4:3"): (2304, 1728),
    ("2K", "16:9"): (2848, 1600),
    ("2K", "9:16"): (1600, 2848),
    ("2K", "3:2"): (2496, 1664),
    ("2K", "2:3"): (1664, 2496),
    ("2K", "21:9"): (3136, 1344),
    # 3K dimensions
    ("3K", "1:1"): (3072, 3072),
    ("3K", "3:4"): (2592, 3456),
    ("3K", "4:3"): (3456, 2592),
    ("3K", "16:9"): (4096, 2304),
    ("3K", "9:16"): (2304, 4096),
    ("3K", "3:2"): (3744, 2496),
    ("3K", "2:3"): (2496, 3744),
    ("3K", "21:9"): (4704, 2016),
}


class OutputFormatEnum(str, Enum):
    """Output image format."""
    png = "png"
    jpeg = "jpeg"


class AppInput(BaseAppInput):
    """Input schema for Seedream 5 Lite image generation."""

    prompt: str = Field(
        description="Text prompt describing the image to generate. Be descriptive about style, composition, and details.",
        examples=["Vibrant close-up editorial portrait, model with piercing gaze, wearing a sculptural hat, rich color blocking"]
    )
    images: Optional[List[File]] = Field(
        default=None,
        description="Optional reference images for image-to-image or multi-image blending. Up to 14 images supported."
    )
    size: SizeEnum = Field(
        default=SizeEnum.size_2k,
        description="Output image resolution. 2K or 3K."
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
    """Output schema for Seedream 5 Lite image generation."""

    image: File = Field(description="The generated image file.")


class App(BaseApp):
    """Seedream 5 Lite image generation application using BytePlus ARK SDK."""

    async def setup(self):
        """Initialize the BytePlus client."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Suppress noisy httpx logs
        logging.getLogger("httpx").setLevel(logging.WARNING)

        # Initialize client
        self.client = setup_byteplus_client()
        self.model_id = "seedream-5-0-260128"

        self.logger.info(f"Seedream 5 Lite initialized with model: {self.model_id}")

    def get_dimensions(self, size: SizeEnum, aspect_ratio: AspectRatioEnum) -> Tuple[int, int]:
        """Get pixel dimensions for size and aspect ratio combination."""
        key = (size.value, aspect_ratio.value)
        return DIMENSIONS.get(key, (2048, 2048))

    async def run(self, input_data: AppInput) -> AppOutput:
        """Generate image using Seedream 5 Lite."""
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
            if input_data.images:
                image_urls = []
                for img in input_data.images:
                    if not img.exists():
                        raise RuntimeError(f"Input image does not exist at path: {img.path}")
                    image_urls.append(img.uri)

                # Single image: pass as string; multiple: pass as list
                image_param = image_urls[0] if len(image_urls) == 1 else image_urls

            # Call image generation API with WIDTHxHEIGHT format
            result = self.client.images.generate(
                model=self.model_id,
                prompt=input_data.prompt,
                size=size_str,
                output_format=input_data.output_format.value,
                response_format="url",
                watermark=input_data.watermark,
                image=image_param,
                sequential_image_generation="disabled",
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

            # Build output metadata for pricing
            output_meta = OutputMeta(
                outputs=[
                    ImageMeta(
                        width=width,
                        height=height,
                        extra={
                            "mode": mode,
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

"""
Seedream 4.5 - BytePlus Image Generation

Generate high-quality images from text prompts with optional image-to-image generation.
Uses BytePlus ARK SDK with synchronous image generation.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field
from typing import Optional
from enum import Enum
import logging
import os

from .byteplus_helper import (
    setup_byteplus_client,
    download_image,
)


class SizeEnum(str, Enum):
    """Image size/resolution options."""
    size_2k = "2K"
    size_4k = "4K"


class AppInput(BaseAppInput):
    """Input schema for Seedream 4.5 image generation."""

    prompt: str = Field(
        description="Text prompt describing the image to generate. Be descriptive about style, composition, and details.",
        examples=["A majestic mountain landscape at sunset with golden light reflecting off snow-capped peaks"]
    )
    image: Optional[File] = Field(
        default=None,
        description="Optional reference image for image-to-image generation. If not provided, generates from text only."
    )
    size: SizeEnum = Field(
        default=SizeEnum.size_2k,
        description="Output image resolution. 2K (2560x1440) or 4K (4096x4096)."
    )
    watermark: bool = Field(
        default=False,
        description="Whether to add a watermark to the generated image."
    )


class AppOutput(BaseAppOutput):
    """Output schema for Seedream 4.5 image generation."""

    image: File = Field(description="The generated image file.")


class App(BaseApp):
    """Seedream 4.5 image generation application using BytePlus ARK SDK."""

    async def setup(self):
        """Initialize the BytePlus client."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Suppress noisy httpx logs
        logging.getLogger("httpx").setLevel(logging.WARNING)

        # Initialize client
        self.client = setup_byteplus_client()
        self.model_id = "seedream-4-5-251128"

        self.logger.info(f"Seedream 4.5 initialized with model: {self.model_id}")

    async def run(self, input_data: AppInput) -> AppOutput:
        """Generate image using Seedream 4.5."""
        try:
            mode = "image-to-image" if input_data.image else "text-to-image"
            self.logger.info(f"Starting {mode} generation")
            self.logger.info(f"Prompt: {input_data.prompt[:100]}...")
            self.logger.info(f"Size: {input_data.size.value}, Watermark: {input_data.watermark}")

            # Build extra body parameters
            extra_body = {
                "response_format": "url",
                "watermark": input_data.watermark,
                "sequential_image_generation": "disabled",
            }

            # Add reference image if provided (I2I mode)
            if input_data.image:
                if not input_data.image.exists():
                    raise RuntimeError(f"Input image does not exist at path: {input_data.image.path}")
                extra_body["image"] = input_data.image.uri

            # Call image generation API (synchronous)
            result = self.client.images.generate(
                model=self.model_id,
                prompt=input_data.prompt,
                size=input_data.size.value,
                extra_body=extra_body,
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

            # Determine dimensions based on size
            width, height = (2560, 1440)  # Default 2K
            if input_data.size == SizeEnum.size_4k:
                width, height = (4096, 4096)

            # Build output metadata for pricing
            output_meta = OutputMeta(
                outputs=[
                    ImageMeta(
                        width=width,
                        height=height,
                        extra={
                            "mode": mode,
                            "watermark": input_data.watermark,
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

"""
Seedream 3.0 T2I - BytePlus Text-to-Image Generation

Generate high-quality images from text prompts.
Uses BytePlus ARK SDK with synchronous image generation.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field
from enum import Enum
import logging
import os

from .byteplus_helper import (
    setup_byteplus_client,
    download_image,
)


class SizeEnum(str, Enum):
    """Image size/resolution options."""
    size_512 = "512x512"
    size_768 = "768x768"
    size_1024 = "1024x1024"
    size_1536 = "1536x1536"
    size_2048 = "2048x2048"
    # Landscape
    size_1024x768 = "1024x768"
    size_1536x1024 = "1536x1024"
    size_2048x1536 = "2048x1536"
    # Portrait
    size_768x1024 = "768x1024"
    size_1024x1536 = "1024x1536"
    size_1536x2048 = "1536x2048"


class AppInput(BaseAppInput):
    """Input schema for Seedream 3.0 T2I image generation."""

    prompt: str = Field(
        description="Text prompt describing the image to generate. Be descriptive about style, composition, and details.",
        examples=["A serene Japanese garden with cherry blossoms in full bloom, soft morning light, cinematic quality"]
    )
    size: SizeEnum = Field(
        default=SizeEnum.size_1024,
        description="Output image resolution. Choose from various square and rectangular aspect ratios up to 2048x2048."
    )
    watermark: bool = Field(
        default=False,
        description="Whether to add a watermark to the generated image."
    )


class AppOutput(BaseAppOutput):
    """Output schema for Seedream 3.0 T2I image generation."""

    image: File = Field(description="The generated image file.")


class App(BaseApp):
    """Seedream 3.0 T2I image generation application using BytePlus ARK SDK."""

    async def setup(self):
        """Initialize the BytePlus client."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Suppress noisy httpx logs
        logging.getLogger("httpx").setLevel(logging.WARNING)

        # Initialize client
        self.client = setup_byteplus_client()
        self.model_id = "seedream-3-0-t2i-250415"

        self.logger.info(f"Seedream 3.0 T2I initialized with model: {self.model_id}")

    async def run(self, input_data: AppInput) -> AppOutput:
        """Generate image using Seedream 3.0 T2I."""
        try:
            self.logger.info("Starting text-to-image generation")
            self.logger.info(f"Prompt: {input_data.prompt[:100]}...")
            self.logger.info(f"Size: {input_data.size.value}, Watermark: {input_data.watermark}")

            # Build extra body parameters
            extra_body = {
                "response_format": "url",
                "watermark": input_data.watermark,
            }

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

            # Parse dimensions from size string
            size_str = input_data.size.value
            parts = size_str.split("x")
            width, height = int(parts[0]), int(parts[1])

            # Build output metadata for pricing
            output_meta = OutputMeta(
                outputs=[
                    ImageMeta(
                        width=width,
                        height=height,
                        extra={
                            "mode": "text-to-image",
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

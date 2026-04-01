"""
Wan 2.7 Image - Alibaba Cloud Image Generation

Generate and edit images using Wan 2.7 Image model via DashScope API.
Supports text-to-image, image editing, interactive editing, and image set generation.
Faster generation speed compared to the Pro variant.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field
from typing import Optional, List
import logging

from .wan_helper import (
    build_message,
    generate_images,
    extract_images,
    download_images,
)


class AppInput(BaseAppInput):
    """Input schema for Wan 2.7 Image."""

    prompt: str = Field(
        description="Text prompt describing what to generate or edit. Supports Chinese and English, up to 5000 characters. For editing, provide reference images."
    )
    reference_images: Optional[List[File]] = Field(
        default=None,
        description="Reference images for editing (0-9 images). Order in array defines image order."
    )
    num_images: int = Field(
        default=1,
        ge=1,
        le=4,
        description="Number of images to generate (1-4). When image set mode is enabled, max is 12."
    )
    size: str = Field(
        default="2K",
        description="Output resolution: '1K' (1024x1024), '2K' (2048x2048, default). Or specify pixel dimensions like '1024*768'."
    )
    watermark: bool = Field(
        default=False,
        description="Add 'AI Generated' watermark to bottom-right corner."
    )
    thinking_mode: bool = Field(
        default=True,
        description="Enable thinking mode for better quality. Only effective for text-to-image without image input or image set mode."
    )
    enable_sequential: bool = Field(
        default=False,
        description="Enable image set output mode for generating consistent multi-image sets (e.g., same character in different scenes)."
    )
    seed: Optional[int] = Field(
        default=None,
        ge=0,
        le=2147483647,
        description="Random seed for reproducibility (0-2147483647). Same seed yields similar outputs."
    )


class AppOutput(BaseAppOutput):
    """Output schema for Wan 2.7 Image."""

    images: List[File] = Field(description="Generated images in PNG format.")


class App(BaseApp):
    """Wan 2.7 Image generation/editing application."""

    async def setup(self, metadata):
        """Initialize the application."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.model = "wan2.7-image"
        self.logger.info("Wan 2.7 Image initialized")

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Generate or edit images using Wan 2.7 Image."""
        try:
            mode = "edit" if input_data.reference_images else "generate"
            self.logger.info(f"Mode: {mode}, generating {input_data.num_images} image(s)")
            self.logger.info(f"Prompt: {input_data.prompt[:100]}...")

            message = build_message(
                prompt=input_data.prompt,
                reference_images=input_data.reference_images,
            )

            # Adjust n limit for sequential mode
            num_images = input_data.num_images
            if input_data.enable_sequential:
                num_images = min(num_images, 12)

            response = generate_images(
                model=self.model,
                message=message,
                num_images=num_images,
                size=input_data.size,
                watermark=input_data.watermark,
                thinking_mode=input_data.thinking_mode,
                enable_sequential=input_data.enable_sequential,
                seed=input_data.seed,
                logger=self.logger,
            )

            image_urls = extract_images(response, logger=self.logger)
            image_paths = download_images(image_urls, logger=self.logger)

            images = [File(path=path) for path in image_paths]

            width, height = 2048, 2048
            if isinstance(response, dict):
                usage = response.get("usage", {})
                size_str = usage.get("size", "")
                if "*" in size_str:
                    parts = size_str.split("*")
                    width, height = int(parts[0]), int(parts[1])

            self.logger.info(f"Successfully generated {len(images)} image(s)")

            return AppOutput(
                images=images,
                output_meta=OutputMeta(
                    outputs=[
                        ImageMeta(
                            width=width,
                            height=height,
                            count=len(images),
                            extra={"mode": mode, "model": "wan2.7-image"},
                        )
                    ]
                ),
            )

        except Exception as e:
            self.logger.error(f"Error during image generation: {e}")
            raise RuntimeError(f"Image generation failed: {str(e)}")

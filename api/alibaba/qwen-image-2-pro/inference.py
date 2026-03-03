"""
Qwen-Image-2.0 Pro - Alibaba Cloud Professional Image Generation

Generate and edit images using Qwen-Image-2.0-Pro model via DashScope API.
Pro series offers enhanced text rendering, fine-grained realism, detailed
photorealistic scenes, and stronger semantic adherence.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field
from typing import Optional, List
import logging

from .alibaba_helper import (
    build_messages,
    generate_images,
    extract_images,
    download_images,
)


class AppInput(BaseAppInput):
    """Input schema for Qwen-Image-2.0 Pro."""

    prompt: str = Field(
        description="Text prompt describing what to generate or edit. Supports up to 800 characters with complex text rendering. For editing, reference images by position (e.g., 'Image 1', 'Image 2')."
    )
    reference_images: Optional[List[File]] = Field(
        default=None,
        description="Reference images for editing (1-3 images). Image 1 can be subject, Image 2 clothing/style, Image 3 pose, etc."
    )
    num_images: int = Field(
        default=1,
        ge=1,
        le=6,
        description="Number of images to generate (1-6)."
    )
    width: Optional[int] = Field(
        default=None,
        ge=512,
        le=2048,
        description="Output width in pixels (512-2048). Total pixels must be between 512*512 and 2048*2048."
    )
    height: Optional[int] = Field(
        default=None,
        ge=512,
        le=2048,
        description="Output height in pixels (512-2048). Total pixels must be between 512*512 and 2048*2048."
    )
    watermark: bool = Field(
        default=False,
        description="Add 'Qwen-Image' watermark to bottom-right corner."
    )
    negative_prompt: str = Field(
        default="",
        description="Content to avoid (e.g., 'low resolution, low quality, deformed limbs'). Max 500 characters."
    )
    prompt_extend: bool = Field(
        default=True,
        description="Enable prompt rewriting for more diverse, detailed content. Disable for precise control over image details."
    )
    seed: Optional[int] = Field(
        default=None,
        ge=0,
        le=2147483647,
        description="Random seed for reproducibility (0-2147483647). Same seed produces more consistent results."
    )


class AppOutput(BaseAppOutput):
    """Output schema for Qwen-Image-2.0 Pro."""

    images: List[File] = Field(description="Generated images in PNG format.")


class App(BaseApp):
    """Qwen-Image-2.0 Pro image generation/editing application."""

    async def setup(self, metadata):
        """Initialize the application."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.model = "qwen-image-2.0-pro"
        self.logger.info(f"Qwen-Image-2.0 Pro initialized")

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Generate or edit images using Qwen-Image-2.0 Pro."""
        try:
            # Determine mode
            mode = "edit" if input_data.reference_images else "generate"
            self.logger.info(f"Mode: {mode}, generating {input_data.num_images} image(s)")
            self.logger.info(f"Prompt: {input_data.prompt[:100]}...")

            # Build messages
            messages = build_messages(
                prompt=input_data.prompt,
                reference_images=input_data.reference_images,
            )

            # Call API
            response = generate_images(
                model=self.model,
                messages=messages,
                num_images=input_data.num_images,
                width=input_data.width,
                height=input_data.height,
                watermark=input_data.watermark,
                negative_prompt=input_data.negative_prompt,
                prompt_extend=input_data.prompt_extend,
                seed=input_data.seed,
                logger=self.logger,
            )

            # Extract and download images
            image_urls = extract_images(response, logger=self.logger)
            image_paths = download_images(image_urls, logger=self.logger)

            # Build output
            images = [File(path=path) for path in image_paths]

            # Get dimensions from response if available
            width, height = 1024, 1024
            if isinstance(response, dict):
                usage = response.get("usage", {})
                width = usage.get("width", 1024)
                height = usage.get("height", 1024)

            output_meta = OutputMeta(
                outputs=[
                    ImageMeta(
                        width=width,
                        height=height,
                        count=len(images),
                        extra={"mode": mode, "model": "pro"},
                    )
                ]
            )

            self.logger.info(f"Successfully generated {len(images)} image(s)")

            return AppOutput(
                images=images,
                output_meta=output_meta,
            )

        except Exception as e:
            self.logger.error(f"Error during image generation: {e}")
            raise RuntimeError(f"Image generation failed: {str(e)}")

"""
Grok Imagine Image - xAI Image Generation

Generate and edit images using xAI's Grok Imagine model.
Supports text-to-image generation, image editing, and multiple output generation.
"""

import base64
import os
import logging
import tempfile
from typing import Optional, Literal

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field
from xai_sdk import Client


AspectRatioType = Literal["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"]


class AppInput(BaseAppInput):
    """Input schema for Grok Imagine image generation."""

    prompt: str = Field(
        description="Text prompt describing the desired image content.",
        examples=["A cat in a tree", "A futuristic cityscape at sunset"]
    )
    image: Optional[File] = Field(
        default=None,
        description="Optional input image for image editing. When provided, the model will edit this image based on the prompt."
    )
    aspect_ratio: AspectRatioType = Field(
        default="1:1",
        description="Aspect ratio of the generated image."
    )
    n: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Number of images to generate (1-10)."
    )


class AppOutput(BaseAppOutput):
    """Output schema for Grok Imagine image generation."""

    images: list[File] = Field(description="The generated image files.")


class App(BaseApp):
    """Grok Imagine image generation application using xAI SDK."""

    async def setup(self, metadata):
        """Initialize the xAI client."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        api_key = os.environ.get("XAI_API_KEY")
        if not api_key:
            raise RuntimeError("XAI_API_KEY environment variable is required")

        self.client = Client(api_key=api_key)
        self.model = "grok-imagine-image-pro"

        self.logger.info(f"Grok Imagine Image Pro initialized with model: {self.model}")

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Generate or edit images using Grok Imagine Pro."""
        try:
            mode = "image-edit" if input_data.image else "text-to-image"
            self.logger.info(f"Starting {mode} generation")
            self.logger.info(f"Prompt: {input_data.prompt[:100]}...")
            self.logger.info(f"Aspect ratio: {input_data.aspect_ratio}, Count: {input_data.n}")

            # Build kwargs for the API call
            kwargs = {
                "model": self.model,
                "prompt": input_data.prompt,
                "image_format": "url",
            }

            # Add aspect ratio
            if input_data.aspect_ratio != "1:1":
                kwargs["aspect_ratio"] = input_data.aspect_ratio

            # Add input image for editing mode
            if input_data.image:
                if not input_data.image.exists():
                    raise RuntimeError(f"Input image does not exist at path: {input_data.image.path}")

                with open(input_data.image.path, "rb") as f:
                    image_bytes = f.read()
                    base64_string = base64.b64encode(image_bytes).decode("utf-8")

                content_type = input_data.image.content_type or "image/jpeg"
                kwargs["image_url"] = f"data:{content_type};base64,{base64_string}"

            # Generate images
            output_images = []
            if input_data.n == 1:
                response = self.client.image.sample(**kwargs)
                output_images.append(self._save_image(response))
            else:
                kwargs["n"] = input_data.n
                responses = self.client.image.sample_batch(**kwargs)
                for response in responses:
                    output_images.append(self._save_image(response))

            if not output_images:
                raise RuntimeError("No images generated")

            # Determine dimensions based on aspect ratio
            aspect_dimensions = {
                "1:1": (1024, 1024),
                "16:9": (1344, 756),
                "9:16": (756, 1344),
                "4:3": (1152, 864),
                "3:4": (864, 1152),
                "3:2": (1248, 832),
                "2:3": (832, 1248),
            }
            width, height = aspect_dimensions.get(input_data.aspect_ratio, (1024, 1024))

            output_meta = OutputMeta(
                outputs=[
                    ImageMeta(
                        width=width,
                        height=height,
                        extra={
                            "mode": mode,
                            "aspect_ratio": input_data.aspect_ratio,
                        }
                    )
                    for _ in output_images
                ]
            )

            self.logger.info(f"Generated {len(output_images)} image(s) successfully")

            return AppOutput(
                images=output_images,
                output_meta=output_meta,
            )

        except Exception as e:
            self.logger.error(f"Error during image generation: {e}")
            raise RuntimeError(f"Image generation failed: {str(e)}")

    def _save_image(self, response) -> File:
        """Save image response to a temporary file."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            # SDK returns image bytes or has a url property
            if hasattr(response, 'image') and response.image:
                f.write(response.image)
            elif hasattr(response, 'url') and response.url:
                import httpx
                img_response = httpx.get(response.url)
                img_response.raise_for_status()
                f.write(img_response.content)
            else:
                raise RuntimeError(f"Unexpected response format: {response}")
            return File(path=f.name)

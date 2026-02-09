"""
Grok Imagine Image - xAI Image Generation

Generate and edit images using xAI's Grok Imagine model.
Supports text-to-image generation, image editing, and multiple output generation.
"""

from typing import Optional

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field

from .xai_helper import (
    AspectRatioAutoType,
    create_xai_client,
    setup_logger,
    resolve_aspect_ratio,
    get_image_dimensions,
    encode_image_base64,
    save_image_from_response,
)


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
    aspect_ratio: AspectRatioAutoType = Field(
        default="1:1",
        description="Aspect ratio of the generated image. Use 'auto' to automatically match the input image's aspect ratio."
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
        self.logger = setup_logger(__name__)
        self.client = create_xai_client()
        self.model = "grok-imagine-image"
        self.logger.info(f"Grok Imagine Image initialized with model: {self.model}")

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Generate or edit images using Grok Imagine."""
        try:
            mode = "image-edit" if input_data.image else "text-to-image"
            self.logger.info(f"Starting {mode} generation")
            self.logger.info(f"Prompt: {input_data.prompt[:100]}...")

            # Resolve aspect ratio (handle "auto")
            aspect_ratio = resolve_aspect_ratio(
                input_data.aspect_ratio,
                input_data.image,
                self.logger,
            )
            self.logger.info(f"Aspect ratio: {aspect_ratio}, Count: {input_data.n}")

            # Build kwargs for the API call
            kwargs = {
                "model": self.model,
                "prompt": input_data.prompt,
                "image_format": "url",
                "aspect_ratio": aspect_ratio,
            }

            # Add input image for editing mode
            if input_data.image:
                kwargs["image_url"] = encode_image_base64(input_data.image)

            # Generate images
            output_images = []
            if input_data.n == 1:
                response = self.client.image.sample(**kwargs)
                output_images.append(save_image_from_response(response))
            else:
                kwargs["n"] = input_data.n
                responses = self.client.image.sample_batch(**kwargs)
                for response in responses:
                    output_images.append(save_image_from_response(response))

            if not output_images:
                raise RuntimeError("No images generated")

            # Determine dimensions based on aspect ratio
            width, height = get_image_dimensions(aspect_ratio)

            output_meta = OutputMeta(
                outputs=[
                    ImageMeta(
                        width=width,
                        height=height,
                        extra={
                            "mode": mode,
                            "aspect_ratio": aspect_ratio,
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

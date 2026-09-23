"""
Grok Imagine Image Pro - xAI Image Generation

Generate and edit images using xAI's Grok Imagine Pro model.
Supports text-to-image generation, image editing, and multiple output generation.
"""

from typing import Optional

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field

from .xai_helper import (
    AspectRatioAutoType,
    XAIError,
    MODERATION_NOTICE,
    collect_images,
    upstream_cost_usd,
    create_xai_client,
    setup_logger,
    resolve_aspect_ratio,
    get_image_dimensions,
    encode_image_base64,
    retry_on_rate_limit,
)


class AppInput(BaseAppInput):
    """Input schema for Grok Imagine Pro image generation."""

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
    """Output schema for Grok Imagine Pro image generation."""

    images: list[File] = Field(description="The generated image files. Excludes images withheld by xAI content moderation.")
    moderated_count: int = Field(default=0, description="Number of generated images withheld by xAI content moderation.")
    notice: Optional[str] = Field(default=None, description="Set when xAI content moderation withheld some or all images.")


class App(BaseApp):
    """Grok Imagine Pro image generation application using xAI SDK."""

    async def setup(self):
        """Initialize the xAI client."""
        self.logger = setup_logger(__name__)
        self.client = create_xai_client()
        self.model = "grok-imagine-image-pro"
        self.logger.info(f"Grok Imagine Image Pro initialized with model: {self.model}")

    async def run(self, input_data: AppInput) -> AppOutput:
        """Generate or edit images using Grok Imagine Pro."""
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

            # Generate images (with 429 retry)
            if input_data.n == 1:
                response = await retry_on_rate_limit(
                    lambda: self.client.image.sample(**kwargs),
                    logger=self.logger,
                )
                responses = [response]
            else:
                kwargs["n"] = input_data.n
                responses = await retry_on_rate_limit(
                    lambda: self.client.image.sample_batch(**kwargs),
                    logger=self.logger,
                )

            # Moderated images are generated and billed by xAI, so they are
            # billed here too: the run succeeds with those images left out.
            output_images, moderated_count = collect_images(responses, self.logger)
            generated_count = len(output_images) + moderated_count
            # Batch responses share one usage record, so the first holds the request's cost.
            cost_usd = upstream_cost_usd(responses[0]) if responses else None
            if generated_count == 0:
                raise RuntimeError("No images generated")

            # Determine dimensions based on aspect ratio
            width, height = get_image_dimensions(aspect_ratio)

            # Pricing reads inputs[0].count and outputs[0].count, so each side is one item.
            output_meta = OutputMeta(
                inputs=[ImageMeta(count=1, extra={"type": "image_input"})] if input_data.image else [],
                outputs=[
                    ImageMeta(
                        width=width,
                        height=height,
                        count=generated_count,
                        extra={
                            "mode": mode,
                            "aspect_ratio": aspect_ratio,
                            "moderated_count": moderated_count,
                            "upstream_cost_usd": cost_usd,
                        }
                    )
                ]
            )

            self.logger.info(f"Generated {generated_count} image(s), {moderated_count} withheld by moderation, xAI cost ${cost_usd}")

            return AppOutput(
                images=output_images,
                moderated_count=moderated_count,
                notice=MODERATION_NOTICE.format(what=f"{moderated_count} of {generated_count} image(s)") if moderated_count else None,
                output_meta=output_meta,
            )

        except XAIError:
            raise
        except Exception as e:
            self.logger.error(f"Error during image generation: {e}")
            raise RuntimeError(f"Image generation failed: {str(e)}")

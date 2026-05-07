"""
Grok Imagine Image Quality - xAI High-Quality Image Generation

Generate and edit images using xAI's Grok Imagine Quality model.
Supports 1K (1024x1024) and 2K (2048x2048) output resolutions,
text-to-image generation, image editing, and multiple output generation.
"""

from typing import Optional, Literal

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field

from .xai_helper import (
    AspectRatioAutoType,
    XAIError,
    ContentModerationError,
    create_xai_client,
    setup_logger,
    resolve_aspect_ratio,
    encode_image_base64,
    save_image_from_response,
    retry_on_rate_limit,
)


ResolutionType = Literal["1k", "2k"]

# Dimension maps for quality model resolutions
QUALITY_DIMENSIONS = {
    "1k": {
        "1:1": (1024, 1024),
        "16:9": (1344, 756),
        "9:16": (756, 1344),
        "4:3": (1152, 864),
        "3:4": (864, 1152),
        "3:2": (1248, 832),
        "2:3": (832, 1248),
        "2:1": (1448, 724),
        "1:2": (724, 1448),
        "19.5:9": (1504, 694),
        "9:19.5": (694, 1504),
        "20:9": (1520, 684),
        "9:20": (684, 1520),
    },
    "2k": {
        "1:1": (2048, 2048),
        "16:9": (2688, 1512),
        "9:16": (1512, 2688),
        "4:3": (2304, 1728),
        "3:4": (1728, 2304),
        "3:2": (2496, 1664),
        "2:3": (1664, 2496),
        "2:1": (2896, 1448),
        "1:2": (1448, 2896),
        "19.5:9": (3008, 1388),
        "9:19.5": (1388, 3008),
        "20:9": (3040, 1368),
        "9:20": (1368, 3040),
    },
}


class AppInput(BaseAppInput):
    """Input schema for Grok Imagine Quality image generation."""

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
    resolution: ResolutionType = Field(
        default="1k",
        description="Output resolution. '1k' = 1024x1024 base ($0.05/image), '2k' = 2048x2048 base ($0.07/image)."
    )
    n: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Number of images to generate (1-10)."
    )


class AppOutput(BaseAppOutput):
    """Output schema for Grok Imagine Quality image generation."""

    images: list[File] = Field(description="The generated image files.")


class App(BaseApp):
    """Grok Imagine Quality image generation application using xAI SDK."""

    async def setup(self):
        """Initialize the xAI client."""
        self.logger = setup_logger(__name__)
        self.client = create_xai_client()
        self.model = "grok-imagine-image-quality"
        self.logger.info(f"Grok Imagine Image Quality initialized with model: {self.model}")

    async def run(self, input_data: AppInput) -> AppOutput:
        """Generate or edit images using Grok Imagine Quality."""
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
            self.logger.info(f"Aspect ratio: {aspect_ratio}, Resolution: {input_data.resolution}, Count: {input_data.n}")

            # Get expected dimensions for metadata
            res_dims = QUALITY_DIMENSIONS[input_data.resolution]
            width, height = res_dims.get(aspect_ratio, res_dims["1:1"])

            # Build kwargs for the API call
            kwargs = {
                "model": self.model,
                "prompt": input_data.prompt,
                "image_format": "url",
                "aspect_ratio": aspect_ratio,
                "resolution": input_data.resolution,
            }

            # Add input image for editing mode
            if input_data.image:
                kwargs["image_url"] = encode_image_base64(input_data.image)

            # Generate images (with 429 retry)
            output_images = []
            if input_data.n == 1:
                response = await retry_on_rate_limit(
                    lambda: self.client.image.sample(**kwargs),
                    logger=self.logger,
                )
                output_images.append(save_image_from_response(response))
            else:
                kwargs["n"] = input_data.n
                responses = await retry_on_rate_limit(
                    lambda: self.client.image.sample_batch(**kwargs),
                    logger=self.logger,
                )
                for response in responses:
                    output_images.append(save_image_from_response(response))

            if not output_images:
                raise RuntimeError("No images generated")

            output_meta = OutputMeta(
                outputs=[
                    ImageMeta(
                        width=width,
                        height=height,
                        extra={
                            "mode": mode,
                            "aspect_ratio": aspect_ratio,
                            "resolution": input_data.resolution,
                        }
                    )
                    for _ in output_images
                ]
            )

            self.logger.info(f"Generated {len(output_images)} image(s) at {input_data.resolution} successfully")

            return AppOutput(
                images=output_images,
                output_meta=output_meta,
            )

        except XAIError:
            raise
        except Exception as e:
            self.logger.error(f"Error during image generation: {e}")
            raise RuntimeError(f"Image generation failed: {str(e)}")

"""
Grok Imagine Image 2.0 - xAI Image Generation and Editing

Generate images from text, or edit up to five source images, with xAI's
grok-imagine-image-2.0. Quality (low/medium) and resolution (1k/2k) set the
per-image price. xAI's "auto" quality is not offered: the response does not
say which quality was served, so a run could not be billed from its inputs.
"""

import logging
from typing import List, Literal, Optional

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
    encode_image_base64,
    retry_on_rate_limit,
)

MODEL = "grok-imagine-image-2.0"
MAX_SOURCE_IMAGES = 5


class AppInput(BaseAppInput):
    """Input schema for Grok Imagine Image 2.0."""

    prompt: str = Field(
        description="What to generate, or how to change the source images. With several source images, refer to them as <IMAGE_0>, <IMAGE_1>, ... in the order given.",
        examples=["A watercolor painting of a lighthouse at dawn"],
    )
    images: List[File] = Field(
        default_factory=list,
        max_length=MAX_SOURCE_IMAGES,
        description="Source images to edit (up to 5, JPEG/PNG/WebP). Leave empty to generate from the prompt alone.",
    )
    quality: Literal["low", "medium"] = Field(
        default="low",
        description="'low' is fast and cheaper; 'medium' spends more compute per image for finer detail.",
    )
    resolution: Literal["1k", "2k"] = Field(
        default="1k",
        description="Output resolution. Price per image: low 1k $0.04, low 2k $0.06, medium 1k $0.06, medium 2k $0.08.",
    )
    aspect_ratio: AspectRatioAutoType = Field(
        default="auto",
        description="Aspect ratio of the output. 'auto' lets the model choose; when editing it follows the first source image.",
    )
    n: int = Field(default=1, ge=1, le=10, description="Number of images to generate (1-10).")


class AppOutput(BaseAppOutput):
    """Output schema for Grok Imagine Image 2.0."""

    images: list[File] = Field(description="The generated images. Excludes images withheld by xAI content moderation.")
    moderated_count: int = Field(default=0, description="Number of generated images withheld by xAI content moderation.")
    notice: Optional[str] = Field(default=None, description="Set when xAI content moderation withheld some or all images.")


def _image_ref(image: File) -> str:
    if image.uri and image.uri.startswith("http"):
        return image.uri
    return encode_image_base64(image)


class App(BaseApp):
    """Grok Imagine Image 2.0 using the xAI SDK."""

    async def setup(self, metadata):
        self.logger = setup_logger(__name__)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        self.client = create_xai_client()
        self.logger.info(f"Grok Imagine Image 2.0 ready model={MODEL}")

    async def run(self, input_data: AppInput) -> AppOutput:
        try:
            mode = "image-edit" if input_data.images else "text-to-image"
            self.logger.info(
                f"{mode}: quality={input_data.quality} resolution={input_data.resolution} "
                f"aspect_ratio={input_data.aspect_ratio} n={input_data.n} sources={len(input_data.images)}"
            )

            kwargs = {
                "model": MODEL,
                "prompt": input_data.prompt,
                "image_format": "url",
                "quality": input_data.quality,
                "resolution": input_data.resolution,
            }
            # Omitted, the server uses its "auto" default (the SDK rejects the literal).
            if input_data.aspect_ratio != "auto":
                kwargs["aspect_ratio"] = input_data.aspect_ratio
            if len(input_data.images) == 1:
                kwargs["image_url"] = _image_ref(input_data.images[0])
            elif input_data.images:
                kwargs["image_urls"] = [_image_ref(img) for img in input_data.images]

            if input_data.n == 1:
                response = await retry_on_rate_limit(
                    lambda: self.client.image.sample(**kwargs), logger=self.logger, in_thread=True,
                )
                responses = [response]
            else:
                responses = await retry_on_rate_limit(
                    lambda: self.client.image.sample_batch(n=input_data.n, **kwargs), logger=self.logger, in_thread=True,
                )

            # Moderated images are generated and billed by xAI, so they are
            # billed here too: the run succeeds with those images left out.
            output_images, moderated_count = collect_images(responses, self.logger)
            generated_count = len(output_images) + moderated_count
            if generated_count == 0:
                raise RuntimeError("No images generated")
            # Batch responses share one usage record, so the first holds the request's cost.
            cost_usd = upstream_cost_usd(responses[0])

            width, height = 0, 0
            if output_images:
                from PIL import Image
                with Image.open(output_images[0].path) as img:
                    width, height = img.size

            # Pricing reads inputs[0].count and outputs[0].count, so each side is one item.
            output_meta = OutputMeta(
                inputs=[ImageMeta(count=len(input_data.images), extra={"type": "image_input"})] if input_data.images else [],
                outputs=[
                    ImageMeta(
                        width=width,
                        height=height,
                        resolution_mp=round(width * height / 1_000_000, 3),
                        count=generated_count,
                        extra={
                            "mode": mode,
                            "quality": input_data.quality,
                            "resolution": input_data.resolution,
                            "moderated_count": moderated_count,
                            "upstream_cost_usd": cost_usd,
                        },
                    )
                ],
            )

            self.logger.info(
                f"Generated {generated_count} image(s) {width}x{height}, {moderated_count} withheld by moderation, "
                f"{len(input_data.images)} source image(s), xAI cost ${cost_usd}"
            )

            return AppOutput(
                images=output_images,
                moderated_count=moderated_count,
                notice=MODERATION_NOTICE.format(what=f"{moderated_count} of {generated_count} image(s)") if moderated_count else None,
                output_meta=output_meta,
            )

        except XAIError:
            raise
        except Exception as e:
            self.logger.error(f"Image generation failed: {e}", exc_info=True)
            raise RuntimeError(f"Image generation failed: {e}")

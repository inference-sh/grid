"""
GPT Image 2 - OpenAI Image Generation

Generate and edit images using OpenAI's gpt-image-2 model.
Supports text-to-image generation, image editing with reference images, and mask-based editing.
"""

from typing import Optional

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field

from .openai_helper import (
    QualityType,
    OutputFormatType,
    create_openai_client,
    setup_logger,
    validate_and_fix_dimensions,
    make_size_string,
    save_base64_image,
)


class AppInput(BaseAppInput):
    """Input schema for GPT Image 2 generation."""

    prompt: str = Field(
        description="Text prompt describing the desired image.",
        examples=["A cat wearing a tiny top hat, oil painting style"],
    )
    images: Optional[list[File]] = Field(
        default=None,
        description="Optional reference image(s) for editing. When a mask is provided, it applies to the first image.",
    )
    mask: Optional[File] = Field(
        default=None,
        description="Optional mask image indicating areas to edit (requires input images). "
        "Transparent areas in the mask indicate where the image should be edited. Applied to the first image.",
    )
    width: int = Field(
        default=1024,
        ge=256,
        le=3840,
        description="Output image width in pixels. Must be a multiple of 16.",
    )
    height: int = Field(
        default=1024,
        ge=256,
        le=3840,
        description="Output image height in pixels. Must be a multiple of 16.",
    )
    quality: QualityType = Field(
        default="auto",
        description="Rendering quality. 'low' for fast drafts, 'high' for final assets.",
    )
    n: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Number of images to generate (1-10).",
    )
    output_format: OutputFormatType = Field(
        default="png",
        description="Output file format.",
    )
    output_compression: Optional[int] = Field(
        default=None,
        ge=0,
        le=100,
        description="Compression level for jpeg/webp (0-100). Ignored for png.",
    )


class AppOutput(BaseAppOutput):
    """Output schema for GPT Image 2 generation."""

    images: list[File] = Field(description="The generated image files.")


class App(BaseApp):
    """GPT Image 2 image generation application."""

    async def setup(self):
        """Initialize the OpenAI client."""
        self.logger = setup_logger(__name__)
        self.client = create_openai_client()
        self.logger.info("GPT Image 2 initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        """Generate or edit images using GPT Image 2."""
        width, height = validate_and_fix_dimensions(
            input_data.width, input_data.height, logger=self.logger
        )

        is_edit = input_data.images is not None and len(input_data.images) > 0
        mode = "edit" if is_edit else "generate"
        size_str = make_size_string(width, height)

        self.logger.info(f"Starting {mode} — prompt: {input_data.prompt[:100]}")
        self.logger.info(
            f"Size: {size_str}, quality: {input_data.quality}, "
            f"n: {input_data.n}, format: {input_data.output_format}"
        )

        if is_edit:
            result = await self._edit(input_data, size_str)
        else:
            result = await self._generate(input_data, size_str)

        # Save images
        output_images: list[File] = []
        for i, image_b64 in enumerate(result):
            path = save_base64_image(image_b64, input_data.output_format)
            output_images.append(File(path=path))
            self.logger.info(f"Saved image {i + 1}/{len(result)}")

        if not output_images:
            raise RuntimeError("No images generated")

        # Read actual dimensions from first output
        out_w, out_h = width, height
        try:
            from PIL import Image as PILImage

            with PILImage.open(output_images[0].path) as img:
                out_w, out_h = img.size
        except Exception:
            pass

        output_meta = OutputMeta(
            outputs=[
                ImageMeta(
                    width=out_w,
                    height=out_h,
                    count=len(output_images),
                    extra={
                        "mode": mode,
                        "quality": input_data.quality,
                        "model": "gpt-image-2",
                    },
                )
            ]
        )

        self.logger.info(f"Generated {len(output_images)} image(s) — {out_w}x{out_h}")

        return AppOutput(
            images=output_images,
            output_meta=output_meta,
        )

    async def _generate(self, input_data: AppInput, size_str: str) -> list[str]:
        """Text-to-image generation via the Images API."""
        kwargs: dict = {
            "model": "gpt-image-2",
            "prompt": input_data.prompt,
            "n": input_data.n,
            "size": size_str,
            "quality": input_data.quality,
            "output_format": input_data.output_format,
        }
        if input_data.output_compression is not None and input_data.output_format != "png":
            kwargs["output_compression"] = input_data.output_compression

        response = await self.client.images.generate(**kwargs)
        return [img.b64_json for img in response.data]

    async def _edit(self, input_data: AppInput, size_str: str) -> list[str]:
        """Image editing via the Images API. Supports multiple reference images."""
        open_files = []
        try:
            image_files = []
            for img in input_data.images:
                f = open(img.path, "rb")
                open_files.append(f)
                image_files.append(f)

            kwargs: dict = {
                "model": "gpt-image-2",
                "prompt": input_data.prompt,
                "image": image_files if len(image_files) > 1 else image_files[0],
                "n": input_data.n,
                "size": size_str,
                "quality": input_data.quality,
            }

            if input_data.mask and input_data.mask.exists():
                mask_f = open(input_data.mask.path, "rb")
                open_files.append(mask_f)
                kwargs["mask"] = mask_f

            response = await self.client.images.edit(**kwargs)
            return [img.b64_json for img in response.data]
        finally:
            for f in open_files:
                f.close()

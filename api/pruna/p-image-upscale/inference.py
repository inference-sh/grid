"""
P-Image-Upscale - AI-powered image upscaling with detail and realism enhancement
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field
from typing import Optional, Literal
import logging

from .pruna_helper import run_prediction, get_generation_url, download_result, upload_file


class AppInput(BaseAppInput):
    """Input schema for P-Image-Upscale."""

    image: File = Field(
        description="Input image to upscale."
    )
    megapixels: int = Field(
        default=4,
        ge=1,
        le=8,
        description="Target resolution in megapixels (1-8). Output is capped at 8 MP."
    )
    output_format: Literal["jpg", "png", "webp"] = Field(
        default="jpg",
        description="Format of the output image."
    )
    output_quality: int = Field(
        default=80,
        ge=0,
        le=100,
        description="Quality when saving the output image (0-100). Not relevant for PNG."
    )
    enhance_details: bool = Field(
        default=False,
        description="Enhance fine textures and small details. May increase contrast and introduce minor deviations."
    )
    enhance_realism: bool = Field(
        default=False,
        description="Improve realism. May deviate more from the original. Recommended for AI-generated images."
    )
    disable_safety_checker: bool = Field(
        default=False,
        description="Disable safety checker for generated images."
    )


class AppOutput(BaseAppOutput):
    """Output schema for P-Image-Upscale."""

    image: File = Field(description="Upscaled image file.")


class App(BaseApp):
    """P-Image-Upscale for AI-powered image upscaling."""

    async def setup(self, metadata):
        """Initialize the application."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.model = "p-image-upscale"
        self.logger.info("P-Image-Upscale initialized")

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Upscale an image using P-Image-Upscale."""
        try:
            self.logger.info(f"Upscaling image to {input_data.megapixels} MP, format={input_data.output_format}")

            # Upload image or use URL directly
            if input_data.image.uri and input_data.image.uri.startswith("http"):
                image_url = input_data.image.uri
            else:
                if not input_data.image.exists():
                    raise RuntimeError(f"Image does not exist: {input_data.image.path}")
                self.logger.info("Uploading image...")
                upload_result = upload_file(input_data.image.path, logger=self.logger)
                image_url = upload_result.get("urls", {}).get("get")
                if not image_url:
                    raise RuntimeError("Failed to get URL for uploaded image")

            # Build request
            request_data = {
                "image": image_url,
                "target": input_data.megapixels,
                "output_format": input_data.output_format,
                "output_quality": input_data.output_quality,
                "enhance_details": input_data.enhance_details,
                "enhance_realism": input_data.enhance_realism,
            }

            if input_data.disable_safety_checker:
                request_data["disable_safety_checker"] = True

            # Run prediction
            result = await run_prediction(
                model=self.model,
                input_data=request_data,
                use_sync=True,
                logger=self.logger,
            )

            # Download result
            generation_url = get_generation_url(result)
            suffix = f".{input_data.output_format}"
            image_path = download_result(generation_url, suffix=suffix, logger=self.logger)

            # Read input dimensions
            from PIL import Image
            with Image.open(input_data.image.path) as img:
                in_w, in_h = img.size

            # Read output dimensions
            with Image.open(image_path) as img:
                out_w, out_h = img.size

            self.logger.info(f"Upscaled {in_w}x{in_h} -> {out_w}x{out_h}")

            output_meta = OutputMeta(
                inputs=[ImageMeta(width=in_w, height=in_h, count=1)],
                outputs=[ImageMeta(width=out_w, height=out_h, count=1)],
            )

            return AppOutput(
                image=File(path=image_path),
                output_meta=output_meta,
            )

        except Exception as e:
            self.logger.error(f"Error: {e}")
            raise RuntimeError(f"Image upscaling failed: {str(e)}")

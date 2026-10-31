"""
Reve - Image Generation, Editing, and Remix

Consolidates:
- fal-ai/reve/text-to-image - Generate from text prompt
- fal-ai/reve/edit - Edit existing image with prompt
- fal-ai/reve/remix - Style remix of existing image
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field
from typing import Optional, List
from enum import Enum
import logging

from .fal_helper import setup_fal_client, run_fal_model, download_image

logging.getLogger("httpx").setLevel(logging.WARNING)


class ModeEnum(str, Enum):
    """Operation mode."""
    auto = "auto"
    edit = "edit"
    remix = "remix"
    text_to_image = "text-to-image"


class OutputFormatEnum(str, Enum):
    """Output format options."""
    png = "png"
    jpeg = "jpeg"
    webp = "webp"


class AppInput(BaseAppInput):
    prompt: str = Field(description="Text prompt for generation or editing")
    image: Optional[File] = Field(
        default=None,
        description="Input image for edit/remix modes. If not provided, uses text-to-image mode."
    )
    mode: ModeEnum = Field(
        default=ModeEnum.auto,
        description="Operation mode: auto (detect from inputs), edit, remix, or text-to-image"
    )
    output_format: OutputFormatEnum = Field(
        default=OutputFormatEnum.png,
        description="Output image format"
    )


class AppOutput(BaseAppOutput):
    images: List[File] = Field(description="Generated/edited images")


class App(BaseApp):
    async def setup(self):
        """Initialize the application."""
        self.logger = logging.getLogger(__name__)
        self.endpoints = {
            "text-to-image": "fal-ai/reve/text-to-image",
            "edit": "fal-ai/reve/edit",
            "remix": "fal-ai/reve/remix",
        }
        self.logger.info("Reve app initialized")

    def _get_mode(self, input_data: AppInput) -> str:
        """Determine operation mode."""
        if input_data.mode != ModeEnum.auto:
            return input_data.mode.value
        # Auto-detect: no image = text-to-image, with image = edit
        if input_data.image:
            return "edit"
        return "text-to-image"

    def _build_request(self, input_data: AppInput, mode: str) -> dict:
        """Build request payload."""
        request = {
            "prompt": input_data.prompt,
            "output_format": input_data.output_format.value,
        }
        if input_data.image and mode in ("edit", "remix"):
            request["image_url"] = input_data.image.uri
        return request

    async def run(self, input_data: AppInput) -> AppOutput:
        """Run image generation/editing."""
        try:
            setup_fal_client()

            mode = self._get_mode(input_data)
            model_id = self.endpoints[mode]

            # Validate: edit/remix need image
            if mode in ("edit", "remix") and not input_data.image:
                raise ValueError(f"Mode '{mode}' requires an input image")

            self.logger.info(f"Mode: {mode}, Endpoint: {model_id}")
            self.logger.info(f"Prompt: {input_data.prompt[:100]}...")

            request_data = self._build_request(input_data, mode)
            result = run_fal_model(model_id, request_data, self.logger)

            # Download generated images
            output_images = []
            for img_data in result.get("images", []):
                img_path = download_image(img_data["url"], self.logger)
                output_images.append(File(path=img_path))

            return AppOutput(
                images=output_images,
                output_meta=OutputMeta(
                    outputs=[ImageMeta(count=len(output_images))]
                )
            )

        except Exception as e:
            self.logger.error(f"Error: {e}")
            raise RuntimeError(f"Reve failed: {str(e)}")

"""
MAI Image 2.5 — Microsoft's photorealistic image generation and editing model.
Supports text-to-image and image editing with fine-grained pixel-level control.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field
from typing import Optional, Literal, List
import logging

from .fal_helper import setup_fal_client, run_fal_model, download_image


ASPECT_RATIO_DIMENSIONS = {
    "auto": (1024, 1024),
    "1:1": (1024, 1024),
    "16:9": (1820, 1024),
    "9:16": (1024, 1820),
    "4:3": (1365, 1024),
    "3:4": (1024, 1365),
    "3:2": (1536, 1024),
    "2:3": (1024, 1536),
}


class AppInput(BaseAppInput):
    prompt: str = Field(
        description="Text prompt describing the desired image (3-5000 characters).",
        min_length=3,
        max_length=5000,
    )
    image: Optional[File] = Field(
        default=None,
        description="Reference image for editing. When provided, the prompt becomes an edit instruction.",
    )
    num_images: int = Field(
        default=1,
        ge=1,
        le=4,
        description="Number of images to generate (1-4).",
    )
    aspect_ratio: Literal["auto", "1:1", "4:3", "3:4", "16:9", "9:16", "3:2", "2:3"] = Field(
        default="auto",
        description="Output aspect ratio.",
    )
    output_format: Literal["png", "jpeg", "webp"] = Field(
        default="png",
        description="Output image format.",
    )


class AppOutput(BaseAppOutput):
    image: File = Field(description="Generated or edited image.")
    description: Optional[str] = Field(default=None, description="Model description of the generated image.")


class App(BaseApp):
    async def setup(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        self.logger.info("MAI Image 2.5 initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        setup_fal_client()

        # Determine if this is edit or create
        is_edit = input_data.image is not None

        if is_edit:
            model_id = "microsoft/mai-image-2.5/edit"
            self.logger.info(f"Editing image: {input_data.prompt[:80]}...")

            if not input_data.image.exists():
                raise RuntimeError(f"Input image does not exist: {input_data.image.path}")

            # Use the file URI if it's a URL, otherwise use the local path
            image_url = input_data.image.uri if (input_data.image.uri and input_data.image.uri.startswith("http")) else input_data.image.path

            request_data = {
                "prompt": input_data.prompt,
                "image_urls": [image_url],
                "num_images": input_data.num_images,
                "aspect_ratio": input_data.aspect_ratio,
                "output_format": input_data.output_format,
            }
        else:
            model_id = "microsoft/mai-image-2.5"
            self.logger.info(f"Creating image: {input_data.prompt[:80]}...")

            request_data = {
                "prompt": input_data.prompt,
                "num_images": input_data.num_images,
                "aspect_ratio": input_data.aspect_ratio,
                "output_format": input_data.output_format,
            }

        result = run_fal_model(model_id, request_data, self.logger)

        images = result.get("images", [])
        if not images:
            raise RuntimeError("No images returned from the model")

        image_data = images[0]
        image_url = image_data["url"]

        suffix = f".{input_data.output_format}"
        image_path = download_image(image_url, self.logger)

        # Get dimensions
        width, height = ASPECT_RATIO_DIMENSIONS.get(input_data.aspect_ratio, (1024, 1024))
        if image_data.get("width"):
            width = image_data["width"]
        if image_data.get("height"):
            height = image_data["height"]

        self.logger.info(f"Image {'edited' if is_edit else 'generated'}: {width}x{height}")

        return AppOutput(
            image=File(path=image_path),
            description=result.get("description"),
            output_meta=OutputMeta(
                outputs=[ImageMeta(width=width, height=height, count=input_data.num_images)]
            ),
        )

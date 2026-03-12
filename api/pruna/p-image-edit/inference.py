"""
P-Image-Edit - Edit and compose multiple images (1-5) with text instructions
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field
from typing import Optional, List
from enum import Enum
import logging

from .pruna_helper import run_prediction, download_image, upload_file


class AspectRatioEnum(str, Enum):
    match_input = "match_input_image"
    square = "1:1"
    landscape = "16:9"
    portrait = "9:16"
    photo_landscape = "4:3"
    photo_portrait = "3:4"
    classic_landscape = "3:2"
    classic_portrait = "2:3"


class AppInput(BaseAppInput):
    """Input schema for P-Image-Edit."""

    prompt: str = Field(
        description="Text instruction describing the desired edit or composition."
    )
    images: List[File] = Field(
        description="1-5 reference images for editing.",
        min_length=1,
        max_length=5,
    )
    turbo: bool = Field(
        default=True,
        description="Fast mode. Set false for complex tasks requiring more precision."
    )
    aspect_ratio: AspectRatioEnum = Field(
        default=AspectRatioEnum.match_input,
        description="Output aspect ratio."
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducible generation."
    )
    disable_safety_checker: bool = Field(
        default=False,
        description="Disable safety checker for generated images."
    )


class AppOutput(BaseAppOutput):
    """Output schema for P-Image-Edit."""

    image: File = Field(description="Edited/composed image file.")
    seed: Optional[int] = Field(default=None, description="Seed used for generation.")


class App(BaseApp):
    """P-Image-Edit for image editing and composition."""

    async def setup(self, metadata):
        """Initialize the application."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.model = "p-image-edit"
        self.logger.info("P-Image-Edit initialized")

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Edit/compose images using P-Image-Edit."""
        try:
            self.logger.info(f"Editing {len(input_data.images)} image(s)")
            self.logger.info(f"Prompt: {input_data.prompt[:100]}...")

            # Upload images and collect URLs
            image_urls = []
            for i, img in enumerate(input_data.images):
                if not img.exists():
                    raise RuntimeError(f"Image {i+1} does not exist: {img.path}")

                # If it's already a URL, use it directly
                if img.uri and img.uri.startswith("http"):
                    image_urls.append(img.uri)
                else:
                    # Upload local file
                    self.logger.info(f"Uploading image {i+1}...")
                    upload_result = upload_file(img.path, logger=self.logger)
                    file_url = upload_result.get("urls", {}).get("get")
                    if not file_url:
                        raise RuntimeError(f"Failed to get URL for uploaded image {i+1}")
                    image_urls.append(file_url)

            # Build request
            request_data = {
                "prompt": input_data.prompt,
                "images": image_urls,
                "turbo": input_data.turbo,
                "aspect_ratio": input_data.aspect_ratio.value,
            }

            if input_data.seed is not None:
                request_data["seed"] = input_data.seed
            if input_data.disable_safety_checker:
                request_data["disable_safety_checker"] = input_data.disable_safety_checker

            # Run prediction
            result = await run_prediction(
                model=self.model,
                input_data=request_data,
                use_sync=True,
                logger=self.logger,
            )

            # Download result
            generation_url = result.get("generation_url")
            if not generation_url:
                raise RuntimeError("No generation_url in response")

            if generation_url.startswith("/"):
                generation_url = f"https://api.pruna.ai{generation_url}"

            image_path = download_image(generation_url, logger=self.logger)

            output_meta = OutputMeta(
                outputs=[ImageMeta(width=1024, height=1024, count=1)]
            )

            self.logger.info("Image edited successfully")

            return AppOutput(
                image=File(path=image_path),
                seed=result.get("seed"),
                output_meta=output_meta,
            )

        except Exception as e:
            self.logger.error(f"Error: {e}")
            raise RuntimeError(f"Image editing failed: {str(e)}")

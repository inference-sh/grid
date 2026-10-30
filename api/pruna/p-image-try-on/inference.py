"""
P-Image-Try-On - Virtual try-on of garment images onto a person photo by Pruna
"""

import logging
from enum import Enum
from typing import List, Optional

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field

from .pruna_helper import create_prediction, poll_prediction_status, _extract_url, get_generation_url, download_result, file_url


class OutputFormatEnum(str, Enum):
    jpg = "jpg"
    png = "png"
    webp = "webp"


class AppInput(BaseAppInput):
    person_image: File = Field(description="Photo of the person to dress.")
    garment_images: List[File] = Field(
        min_length=1,
        max_length=11,
        description="Garment reference images (up to 6 recommended, 11 supported).",
    )
    prompt: Optional[str] = Field(
        default=None,
        description="Experimental. For non-flatlay garment photos, which garment to take from which image, e.g. 'the green t-shirt from image 1'.",
    )
    reference_pose: Optional[File] = Field(
        default=None,
        description="Optional pose image. The person is reposed to match it before the try-on.",
    )
    turbo: bool = Field(default=False, description="Run faster. Not recommended for more than 4 garments.")
    preserve_input_size: bool = Field(default=True, description="Return the output at the person image's resolution.")
    output_format: OutputFormatEnum = Field(default=OutputFormatEnum.jpg, description="Output image format.")
    output_quality: int = Field(default=95, ge=0, le=100, description="Quality for jpg/webp outputs (0-100).")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducible generation.")


class AppOutput(BaseAppOutput):
    image: File = Field(description="Person wearing the garments.")
    seed: Optional[int] = Field(default=None, description="Seed used for generation.")


class App(BaseApp):
    async def setup(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.model = "p-image-try-on"

    async def run(self, input_data: AppInput) -> AppOutput:
        request_data = {
            "person_image": file_url(input_data.person_image, logger=self.logger),
            "garment_images": [file_url(g, logger=self.logger) for g in input_data.garment_images],
            "turbo": input_data.turbo,
            "preserve_input_size": input_data.preserve_input_size,
            "output_format": input_data.output_format.value,
            "output_quality": input_data.output_quality,
        }
        if input_data.prompt:
            request_data["prompt"] = input_data.prompt
        if input_data.reference_pose:
            request_data["reference_pose"] = file_url(input_data.reference_pose, logger=self.logger)
        if input_data.seed is not None:
            request_data["seed"] = input_data.seed

        self.logger.info("p-image-try-on: %d garment(s), pose=%s, turbo=%s",
                         len(input_data.garment_images), bool(input_data.reference_pose), input_data.turbo)

        # Image model: try sync first, fall back to polling
        result = create_prediction(model=self.model, input_data=request_data, try_sync=True, logger=self.logger)
        if not _extract_url(result):
            if not result.get("id"):
                raise RuntimeError("No prediction ID in response")
            result = await poll_prediction_status(result["id"], logger=self.logger)

        self.logger.info("Prediction metadata: %s", result.get("metadata"))
        image_path = download_result(get_generation_url(result), suffix="." + input_data.output_format.value, logger=self.logger)

        from PIL import Image
        with Image.open(image_path) as img:
            width, height = img.size

        # Pruna prices per image input: the person photo, each garment and the pose image
        image_inputs = 1 + len(input_data.garment_images) + int(input_data.reference_pose is not None)
        output_meta = OutputMeta(
            inputs=[ImageMeta(count=image_inputs)],
            outputs=[ImageMeta(width=width, height=height, count=1)],
        )
        self.logger.info("Try-on image: %sx%s, %d image inputs", width, height, image_inputs)
        return AppOutput(image=File(path=image_path), seed=result.get("seed"), output_meta=output_meta)

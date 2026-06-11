"""
Reve Remix — Create images from text and 1-6 reference images.
Combine multiple references into a single output guided by a prompt.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field
from typing import Optional, Literal, List
import logging

from .reve_helper import remix_image, save_base64_image, image_to_base64, get_api_key


class AppInput(BaseAppInput):
    images: List[File] = Field(
        description="Reference images to remix (1-6). Reference them in the prompt as '0', '1', etc.",
        min_length=1,
        max_length=6,
    )
    prompt: str = Field(
        description="Describe how to combine the references (e.g. 'The woman from 0 driving the car from 1').",
        max_length=2560,
    )
    aspect_ratio: Optional[Literal["16:9", "9:16", "3:2", "2:3", "4:3", "3:4", "1:1"]] = Field(
        default=None,
        description="Output aspect ratio. If not set, chosen by the model.",
    )
    test_time_scaling: Optional[int] = Field(
        default=None,
        ge=1,
        le=5,
        description="Spend more time for better quality (1-5). Values above 1 cost additional credits.",
    )
    upscale_factor: Optional[int] = Field(
        default=None,
        ge=2,
        le=4,
        description="Upscale output by 2x, 3x, or 4x.",
    )
    remove_background: bool = Field(
        default=False,
        description="Make background transparent.",
    )


class AppOutput(BaseAppOutput):
    image: File = Field(description="Remixed image.")


class App(BaseApp):
    async def setup(self):
        self.logger = logging.getLogger(__name__)
        get_api_key()
        self.logger.info("Reve Remix initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        self.logger.info(f"Remixing {len(input_data.images)} image(s): {input_data.prompt[:80]}...")

        ref_images_b64 = []
        for i, img in enumerate(input_data.images):
            if not img.exists():
                raise RuntimeError(f"Reference image {i} does not exist: {img.path}")
            b64 = image_to_base64(img.path)
            self.logger.info(f"Reference image {i} encoded: {len(b64)} chars")
            ref_images_b64.append(b64)

        postprocessing = []
        if input_data.upscale_factor:
            postprocessing.append({"process": "upscale", "upscale_factor": input_data.upscale_factor})
        if input_data.remove_background:
            postprocessing.append({"process": "remove_background"})

        result = remix_image(
            prompt=input_data.prompt,
            reference_images_b64=ref_images_b64,
            aspect_ratio=input_data.aspect_ratio,
            test_time_scaling=input_data.test_time_scaling,
            postprocessing=postprocessing or None,
            logger=self.logger,
        )

        if result.get("content_violation"):
            raise RuntimeError("Content policy violation detected")

        image_path = save_base64_image(result["image"], logger=self.logger)

        from PIL import Image
        with Image.open(image_path) as img:
            width, height = img.size

        self.logger.info(f"Image remixed: {width}x{height}")

        return AppOutput(
            image=File(path=image_path),
            output_meta=OutputMeta(
                outputs=[ImageMeta(width=width, height=height, count=1)]
            ),
        )

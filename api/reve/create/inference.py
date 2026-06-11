"""
Reve Create — Generate images from text prompts.
Best-in-class prompt adherence and text rendering via TypoGuard.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field
from typing import Optional, Literal
import logging

from .reve_helper import create_image, save_base64_image, get_api_key


class AppInput(BaseAppInput):
    prompt: str = Field(
        description="Text description of the image to generate (max 2560 characters).",
        max_length=2560,
    )
    aspect_ratio: Literal["16:9", "9:16", "3:2", "2:3", "4:3", "3:4", "1:1"] = Field(
        default="1:1",
        description="Output aspect ratio.",
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
    image: File = Field(description="Generated image.")


class App(BaseApp):
    async def setup(self):
        self.logger = logging.getLogger(__name__)
        get_api_key()
        self.logger.info("Reve Create initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        self.logger.info(f"Creating image: {input_data.prompt[:80]}...")

        postprocessing = []
        if input_data.upscale_factor:
            postprocessing.append({"process": "upscale", "upscale_factor": input_data.upscale_factor})
        if input_data.remove_background:
            postprocessing.append({"process": "remove_background"})

        result = create_image(
            prompt=input_data.prompt,
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

        self.logger.info(f"Image generated: {width}x{height}")

        return AppOutput(
            image=File(path=image_path),
            output_meta=OutputMeta(
                outputs=[ImageMeta(width=width, height=height, count=1)]
            ),
        )

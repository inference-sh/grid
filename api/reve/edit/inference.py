"""
Reve Edit — Edit images with natural language instructions.
Top 3 on LMArena and Artificial Analysis leaderboards for image editing.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field
from typing import Optional, Literal
import logging

from .reve_helper import edit_image, save_base64_image, image_to_base64, get_api_key


class AppInput(BaseAppInput):
    image: File = Field(
        description="Reference image to edit.",
    )
    edit_instruction: str = Field(
        description="Natural language instruction for the edit (e.g. 'Remove all people in the background').",
        max_length=2560,
    )
    aspect_ratio: Optional[Literal["16:9", "9:16", "3:2", "2:3", "4:3", "3:4", "1:1"]] = Field(
        default=None,
        description="Output aspect ratio. If not set, uses the reference image's ratio.",
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
    image: File = Field(description="Edited image.")


class App(BaseApp):
    async def setup(self):
        self.logger = logging.getLogger(__name__)
        get_api_key()
        self.logger.info("Reve Edit initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        self.logger.info(f"Editing image: {input_data.edit_instruction[:80]}...")

        if not input_data.image.exists():
            raise RuntimeError(f"Input image does not exist: {input_data.image.path}")

        ref_b64 = image_to_base64(input_data.image.path)
        self.logger.info(f"Reference image encoded: {len(ref_b64)} chars")

        postprocessing = []
        if input_data.upscale_factor:
            postprocessing.append({"process": "upscale", "upscale_factor": input_data.upscale_factor})
        if input_data.remove_background:
            postprocessing.append({"process": "remove_background"})

        result = edit_image(
            edit_instruction=input_data.edit_instruction,
            reference_image_b64=ref_b64,
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

        self.logger.info(f"Image edited: {width}x{height}")

        return AppOutput(
            image=File(path=image_path),
            output_meta=OutputMeta(
                outputs=[ImageMeta(width=width, height=height, count=1)]
            ),
        )

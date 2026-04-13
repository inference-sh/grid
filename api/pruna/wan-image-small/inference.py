"""
Wan-Image-Small - Fast, efficient text-to-image generation
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field
from typing import Optional, List
from enum import Enum
import logging

from .pruna_helper import run_prediction, get_generation_url, download_image


class AspectRatioEnum(str, Enum):
    square = "1:1"
    landscape_16_9 = "16:9"
    portrait_9_16 = "9:16"
    landscape_4_3 = "4:3"
    portrait_3_4 = "3:4"
    ultrawide = "21:9"
    custom = "custom"


class AppInput(BaseAppInput):
    prompt: str = Field(description="Text description of the image to generate.")
    aspect_ratio: AspectRatioEnum = Field(default=AspectRatioEnum.landscape_16_9, description="Image aspect ratio.")
    width: Optional[int] = Field(default=None, description="Width (only when aspect_ratio=custom, multiple of 16).")
    height: Optional[int] = Field(default=None, description="Height (only when aspect_ratio=custom, multiple of 16).")
    juiced: bool = Field(default=False, description="Enable faster generation mode.")
    num_outputs: int = Field(default=1, ge=1, le=4, description="Number of images to generate.")
    seed: Optional[int] = Field(default=None, description="Random seed.")


class AppOutput(BaseAppOutput):
    images: List[File] = Field(description="Generated image files.")


class App(BaseApp):
    async def setup(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.model = "wan-image-small"

    async def run(self, input_data: AppInput) -> AppOutput:
        try:
            self.logger.info(f"Generating: {input_data.prompt[:100]}...")
            request_data = {
                "prompt": input_data.prompt,
                "aspect_ratio": input_data.aspect_ratio.value,
                "juiced": input_data.juiced,
                "num_outputs": input_data.num_outputs,
            }
            if input_data.aspect_ratio == AspectRatioEnum.custom:
                if input_data.width:
                    request_data["width"] = input_data.width
                if input_data.height:
                    request_data["height"] = input_data.height
            if input_data.seed is not None:
                request_data["seed"] = input_data.seed

            result = await run_prediction(model=self.model, input_data=request_data, use_sync=True, logger=self.logger)
            generation_url = get_generation_url(result)

            image_path = download_image(generation_url, logger=self.logger)

            # Calculate dimensions from aspect ratio or custom size
            if input_data.aspect_ratio == AspectRatioEnum.custom and input_data.width and input_data.height:
                width, height = input_data.width, input_data.height
            else:
                base_size = 1024
                aspect_ratios = {
                    "1:1": (1, 1), "16:9": (16, 9), "9:16": (9, 16),
                    "4:3": (4, 3), "3:4": (3, 4), "21:9": (21, 9),
                }
                w_ratio, h_ratio = aspect_ratios.get(input_data.aspect_ratio.value, (1, 1))
                if w_ratio >= h_ratio:
                    width = base_size
                    height = int(base_size * h_ratio / w_ratio)
                else:
                    height = base_size
                    width = int(base_size * w_ratio / h_ratio)

            return AppOutput(images=[File(path=image_path)], output_meta=OutputMeta(outputs=[ImageMeta(width=width, height=height, count=input_data.num_outputs)]))
        except Exception as e:
            self.logger.error(f"Error: {e}")
            raise RuntimeError(f"Generation failed: {str(e)}")

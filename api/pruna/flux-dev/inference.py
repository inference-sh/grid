"""
Flux Dev - Advanced text-to-image generation with speed optimizations
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field
from typing import Optional
from enum import Enum
import logging

from .pruna_helper import run_prediction, get_generation_url, download_image


class SpeedModeEnum(str, Enum):
    lightly_juiced = "Lightly Juiced 🍊 (more consistent)"
    juiced = "Juiced 🔥 (default)"
    extra_juiced = "Extra Juiced 🔥 (more speed)"
    blink = "Blink of an eye 👁️"


class AspectRatioEnum(str, Enum):
    square = "1:1"
    landscape_16_9 = "16:9"
    ultrawide = "21:9"
    photo_landscape = "3:2"
    photo_portrait = "2:3"
    portrait_4_5 = "4:5"
    landscape_5_4 = "5:4"
    portrait_3_4 = "3:4"
    landscape_4_3 = "4:3"
    portrait_9_16 = "9:16"
    portrait_9_21 = "9:21"


class OutputFormatEnum(str, Enum):
    jpg = "jpg"
    png = "png"
    webp = "webp"


class AppInput(BaseAppInput):
    prompt: str = Field(description="Text description of the image to generate.")
    aspect_ratio: AspectRatioEnum = Field(default=AspectRatioEnum.square, description="Aspect ratio of output.")
    speed_mode: SpeedModeEnum = Field(default=SpeedModeEnum.extra_juiced, description="Speed optimization level.")
    num_inference_steps: int = Field(default=28, ge=1, le=50, description="Number of inference steps.")
    guidance: float = Field(default=3.5, ge=0, le=10, description="How closely to follow the prompt.")
    seed: Optional[int] = Field(default=None, description="Random seed (-1 for random).")
    image_size: int = Field(default=1024, description="Base size for longest side.")
    output_format: OutputFormatEnum = Field(default=OutputFormatEnum.jpg, description="Output format.")
    output_quality: int = Field(default=80, ge=1, le=100, description="Quality for jpg/webp.")


class AppOutput(BaseAppOutput):
    image: File = Field(description="Generated image file.")


class App(BaseApp):
    async def setup(self, metadata):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.model = "flux-dev"

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        try:
            self.logger.info(f"Generating: {input_data.prompt[:100]}...")
            request_data = {
                "prompt": input_data.prompt,
                "aspect_ratio": input_data.aspect_ratio.value,
                "speed_mode": input_data.speed_mode.value,
                "num_inference_steps": input_data.num_inference_steps,
                "guidance": input_data.guidance,
                "image_size": input_data.image_size,
                "output_format": input_data.output_format.value,
                "output_quality": input_data.output_quality,
            }
            if input_data.seed is not None:
                request_data["seed"] = input_data.seed

            result = await run_prediction(model=self.model, input_data=request_data, use_sync=True, logger=self.logger)
            generation_url = get_generation_url(result)

            image_path = download_image(generation_url, logger=self.logger)

            # Calculate actual dimensions from aspect ratio and image_size
            aspect_ratios = {
                "1:1": (1, 1), "16:9": (16, 9), "9:16": (9, 16),
                "21:9": (21, 9), "9:21": (9, 21), "3:2": (3, 2), "2:3": (2, 3),
                "4:5": (4, 5), "5:4": (5, 4), "3:4": (3, 4), "4:3": (4, 3),
            }
            w_ratio, h_ratio = aspect_ratios.get(input_data.aspect_ratio.value, (1, 1))
            if w_ratio >= h_ratio:
                width = input_data.image_size
                height = int(input_data.image_size * h_ratio / w_ratio)
            else:
                height = input_data.image_size
                width = int(input_data.image_size * w_ratio / h_ratio)

            return AppOutput(image=File(path=image_path), output_meta=OutputMeta(outputs=[ImageMeta(width=width, height=height, count=1)]))
        except Exception as e:
            self.logger.error(f"Error: {e}")
            raise RuntimeError(f"Generation failed: {str(e)}")

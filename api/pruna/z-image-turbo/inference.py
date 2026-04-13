"""
Z-Image-Turbo - Fast image generation
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field
from typing import Optional
from enum import Enum
import logging

from .pruna_helper import run_prediction, get_generation_url, download_image


class OutputFormatEnum(str, Enum):
    jpg = "jpg"
    png = "png"
    webp = "webp"


class AppInput(BaseAppInput):
    prompt: str = Field(description="Text prompt for image generation.")
    width: int = Field(default=1024, ge=64, le=2048, description="Width of the generated image.")
    height: int = Field(default=1024, ge=64, le=2048, description="Height of the generated image.")
    num_inference_steps: int = Field(default=8, ge=1, le=50, description="Number of inference steps.")
    guidance_scale: float = Field(default=0.0, ge=0, le=20, description="Guidance scale (0 for Turbo models).")
    go_fast: bool = Field(default=False, description="Apply additional optimizations.")
    seed: Optional[int] = Field(default=None, description="Random seed.")
    output_format: OutputFormatEnum = Field(default=OutputFormatEnum.jpg, description="Output format.")
    output_quality: int = Field(default=80, ge=0, le=100, description="Quality for jpg/webp.")


class AppOutput(BaseAppOutput):
    image: File = Field(description="Generated image file.")


class App(BaseApp):
    async def setup(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.model = "z-image-turbo"

    async def run(self, input_data: AppInput) -> AppOutput:
        try:
            self.logger.info(f"Generating: {input_data.prompt[:100]}...")
            request_data = {
                "prompt": input_data.prompt,
                "width": input_data.width,
                "height": input_data.height,
                "num_inference_steps": input_data.num_inference_steps,
                "guidance_scale": input_data.guidance_scale,
                "go_fast": input_data.go_fast,
                "output_format": input_data.output_format.value,
                "output_quality": input_data.output_quality,
            }
            if input_data.seed is not None:
                request_data["seed"] = input_data.seed

            result = await run_prediction(model=self.model, input_data=request_data, use_sync=True, logger=self.logger)
            generation_url = get_generation_url(result)

            image_path = download_image(generation_url, logger=self.logger)
            return AppOutput(image=File(path=image_path), output_meta=OutputMeta(outputs=[ImageMeta(width=input_data.width, height=input_data.height, count=1)]))
        except Exception as e:
            self.logger.error(f"Error: {e}")
            raise RuntimeError(f"Generation failed: {str(e)}")

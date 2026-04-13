"""
Qwen-Image - Advanced text-to-image generation with LoRA and prompt enhancement
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field
from typing import Optional, List
from enum import Enum
import logging

from .pruna_helper import run_prediction, get_generation_url, download_image, upload_file


class AspectRatioEnum(str, Enum):
    square = "1:1"
    landscape_16_9 = "16:9"
    portrait_9_16 = "9:16"
    landscape_4_3 = "4:3"
    portrait_3_4 = "3:4"
    photo_landscape = "3:2"
    photo_portrait = "2:3"


class ImageSizeEnum(str, Enum):
    quality = "optimize_for_quality"
    speed = "optimize_for_speed"


class OutputFormatEnum(str, Enum):
    webp = "webp"
    jpg = "jpg"
    png = "png"


class AppInput(BaseAppInput):
    prompt: str = Field(description="Text description of the image to generate.")
    enhance_prompt: bool = Field(default=False, description="Auto-enhance prompt for better results.")
    go_fast: bool = Field(default=True, description="Run faster with optimizations.")
    guidance: float = Field(default=3.0, ge=0, le=10, description="How closely to follow the prompt.")
    negative_prompt: str = Field(default="", description="Things to avoid (e.g., 'blurry, low quality').")
    num_inference_steps: int = Field(default=30, ge=1, le=50, description="Number of denoising steps.")
    seed: Optional[int] = Field(default=None, description="Random seed.")
    disable_safety_checker: bool = Field(default=False, description="Disable safety checker.")
    image: Optional[File] = Field(default=None, description="Input image for img2img mode.")
    strength: float = Field(default=0.9, ge=0, le=1, description="Strength for img2img.")
    lora_weights: Optional[str] = Field(default=None, description="URL to LoRA weights file.")
    lora_scale: float = Field(default=1.0, description="LoRA application strength.")
    aspect_ratio: AspectRatioEnum = Field(default=AspectRatioEnum.landscape_16_9, description="Aspect ratio.")
    image_size: ImageSizeEnum = Field(default=ImageSizeEnum.quality, description="Optimize for quality or speed.")
    output_format: OutputFormatEnum = Field(default=OutputFormatEnum.webp, description="Output format.")
    output_quality: int = Field(default=80, ge=0, le=100, description="Quality for jpg/webp.")


class AppOutput(BaseAppOutput):
    image: File = Field(description="Generated image file.")


class App(BaseApp):
    async def setup(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.model = "qwen-image"

    async def run(self, input_data: AppInput) -> AppOutput:
        try:
            self.logger.info(f"Generating: {input_data.prompt[:100]}...")
            request_data = {
                "prompt": input_data.prompt,
                "enhance_prompt": input_data.enhance_prompt,
                "go_fast": input_data.go_fast,
                "guidance": input_data.guidance,
                "num_inference_steps": input_data.num_inference_steps,
                "aspect_ratio": input_data.aspect_ratio.value,
                "image_size": input_data.image_size.value,
                "output_format": input_data.output_format.value,
                "output_quality": input_data.output_quality,
                "disable_safety_checker": input_data.disable_safety_checker,
            }
            if input_data.negative_prompt:
                request_data["negative_prompt"] = input_data.negative_prompt
            if input_data.seed is not None:
                request_data["seed"] = input_data.seed
            if input_data.lora_weights:
                request_data["lora_weights"] = input_data.lora_weights
                request_data["lora_scale"] = input_data.lora_scale
            if input_data.image:
                if input_data.image.uri and input_data.image.uri.startswith("http"):
                    request_data["image"] = input_data.image.uri
                else:
                    upload_result = upload_file(input_data.image.path, logger=self.logger)
                    request_data["image"] = upload_result.get("urls", {}).get("get")
                request_data["strength"] = input_data.strength

            result = await run_prediction(model=self.model, input_data=request_data, use_sync=True, logger=self.logger)
            generation_url = get_generation_url(result)

            image_path = download_image(generation_url, logger=self.logger)

            # Read input image dimensions if provided
            input_metas = []
            if input_data.image and input_data.image.path:
                from PIL import Image
                with Image.open(input_data.image.path) as pil_img:
                    in_w, in_h = pil_img.size
                input_metas.append(ImageMeta(width=in_w, height=in_h, count=1))

            # Calculate dimensions from aspect ratio
            base_size = 1024
            aspect_ratios = {
                "1:1": (1, 1), "16:9": (16, 9), "9:16": (9, 16),
                "4:3": (4, 3), "3:4": (3, 4), "3:2": (3, 2), "2:3": (2, 3),
            }
            w_ratio, h_ratio = aspect_ratios.get(input_data.aspect_ratio.value, (1, 1))
            if w_ratio >= h_ratio:
                width = base_size
                height = int(base_size * h_ratio / w_ratio)
            else:
                height = base_size
                width = int(base_size * w_ratio / h_ratio)

            return AppOutput(image=File(path=image_path), output_meta=OutputMeta(inputs=input_metas, outputs=[ImageMeta(width=width, height=height, count=1)]))
        except Exception as e:
            self.logger.error(f"Error: {e}")
            raise RuntimeError(f"Generation failed: {str(e)}")

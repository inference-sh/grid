"""
Flux Dev LoRA - Text-to-image and image-to-image with custom LoRA weights
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field
from typing import Optional, List
from enum import Enum
import logging

from .pruna_helper import run_prediction, get_generation_url, download_image, upload_file


class SpeedModeEnum(str, Enum):
    base = "Base Model (compiled)"
    lightly_juiced = "Lightly Juiced 🍊"
    juiced = "Juiced 🧃"
    extra_juiced = "Extra Juiced 🔥"


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


class MegapixelsEnum(str, Enum):
    one = "1"
    quarter = "0.25"


class OutputFormatEnum(str, Enum):
    jpg = "jpg"
    png = "png"
    webp = "webp"


class AppInput(BaseAppInput):
    prompt: str = Field(description="Text description of the desired output.")
    lora: Optional[str] = Field(default=None, description="HuggingFace LoRA URL (e.g., 'owner/model-name').")
    lora_scale: float = Field(default=1.0, ge=-1, le=3, description="LoRA application strength.")
    extra_lora: Optional[str] = Field(default=None, description="Second LoRA URL for combining styles.")
    extra_lora_scale: float = Field(default=1.0, ge=-1, le=3, description="Second LoRA strength.")
    image: Optional[File] = Field(default=None, description="Input image for img2img mode.")
    prompt_strength: float = Field(default=0.8, ge=0, le=1, description="How much to transform input image.")
    num_outputs: int = Field(default=1, ge=1, le=4, description="Number of images to generate.")
    num_inference_steps: int = Field(default=28, ge=1, le=50, description="Number of denoising steps.")
    guidance: float = Field(default=3.0, ge=0, le=10, description="Guidance scale.")
    seed: Optional[int] = Field(default=None, description="Random seed.")
    aspect_ratio: AspectRatioEnum = Field(default=AspectRatioEnum.square, description="Aspect ratio.")
    megapixels: MegapixelsEnum = Field(default=MegapixelsEnum.one, description="Output resolution in megapixels.")
    speed_mode: SpeedModeEnum = Field(default=SpeedModeEnum.juiced, description="Speed optimization.")
    output_format: OutputFormatEnum = Field(default=OutputFormatEnum.jpg, description="Output format.")
    output_quality: int = Field(default=80, ge=1, le=100, description="Quality for jpg/webp.")


class AppOutput(BaseAppOutput):
    images: List[File] = Field(description="Generated image files.")


class App(BaseApp):
    async def setup(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.model = "flux-dev-lora"

    async def run(self, input_data: AppInput) -> AppOutput:
        try:
            self.logger.info(f"Generating with LoRA: {input_data.lora}")
            request_data = {
                "prompt": input_data.prompt,
                "lora_scale": input_data.lora_scale,
                "num_outputs": input_data.num_outputs,
                "num_inference_steps": input_data.num_inference_steps,
                "guidance": input_data.guidance,
                "aspect_ratio": input_data.aspect_ratio.value,
                "megapixels": input_data.megapixels.value,
                "output_format": input_data.output_format.value,
                "output_quality": input_data.output_quality,
                "speed_mode": input_data.speed_mode.value,
            }
            if input_data.lora:
                request_data["lora"] = input_data.lora
            if input_data.extra_lora:
                request_data["extra_lora"] = input_data.extra_lora
                request_data["extra_lora_scale"] = input_data.extra_lora_scale
            if input_data.seed is not None:
                request_data["seed"] = input_data.seed
            if input_data.image:
                if input_data.image.uri and input_data.image.uri.startswith("http"):
                    request_data["image"] = input_data.image.uri
                else:
                    upload_result = upload_file(input_data.image.path, logger=self.logger)
                    request_data["image"] = upload_result.get("urls", {}).get("get")
                request_data["prompt_strength"] = input_data.prompt_strength

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

            # Calculate dimensions from aspect ratio and megapixels
            base_size = 1024 if input_data.megapixels.value == "1" else 512
            aspect_ratios = {
                "1:1": (1, 1), "16:9": (16, 9), "9:16": (9, 16),
                "21:9": (21, 9), "9:21": (9, 21), "3:2": (3, 2), "2:3": (2, 3),
                "4:5": (4, 5), "5:4": (5, 4), "3:4": (3, 4), "4:3": (4, 3),
            }
            w_ratio, h_ratio = aspect_ratios.get(input_data.aspect_ratio.value, (1, 1))
            if w_ratio >= h_ratio:
                width = base_size
                height = int(base_size * h_ratio / w_ratio)
            else:
                height = base_size
                width = int(base_size * w_ratio / h_ratio)

            return AppOutput(images=[File(path=image_path)], output_meta=OutputMeta(inputs=input_metas, outputs=[ImageMeta(width=width, height=height, count=input_data.num_outputs)]))
        except Exception as e:
            self.logger.error(f"Error: {e}")
            raise RuntimeError(f"Generation failed: {str(e)}")

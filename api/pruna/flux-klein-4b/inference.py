"""
Flux Klein 4B - Lightweight, efficient text-to-image generation
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
    ultrawide = "21:9"
    photo_landscape = "3:2"
    photo_portrait = "2:3"
    portrait_4_5 = "4:5"
    landscape_5_4 = "5:4"
    portrait_3_4 = "3:4"
    landscape_4_3 = "4:3"
    portrait_9_16 = "9:16"
    portrait_9_21 = "9:21"
    match_input = "match_input_image"


class OutputFormatEnum(str, Enum):
    jpg = "jpg"
    png = "png"
    webp = "webp"


class MegapixelsEnum(str, Enum):
    quarter = "0.25"
    half = "0.5"
    one = "1"
    two = "2"
    four = "4"


class AppInput(BaseAppInput):
    prompt: str = Field(description="Text description of the image to generate.")
    aspect_ratio: AspectRatioEnum = Field(default=AspectRatioEnum.square, description="Aspect ratio.")
    images: Optional[List[File]] = Field(default=None, description="Input images for img2img (max 5).")
    output_megapixels: MegapixelsEnum = Field(default=MegapixelsEnum.one, description="Output resolution in megapixels.")
    go_fast: bool = Field(default=False, description="Run faster with optimizations.")
    seed: Optional[int] = Field(default=None, description="Random seed.")
    output_format: OutputFormatEnum = Field(default=OutputFormatEnum.jpg, description="Output format.")
    output_quality: int = Field(default=95, ge=0, le=100, description="Quality for jpg/webp.")
    disable_safety_checker: bool = Field(default=False, description="Disable safety checker.")


class AppOutput(BaseAppOutput):
    image: File = Field(description="Generated image file.")


class App(BaseApp):
    async def setup(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.model = "flux-klein-4b"

    async def run(self, input_data: AppInput) -> AppOutput:
        try:
            self.logger.info(f"Generating: {input_data.prompt[:100]}...")
            request_data = {
                "prompt": input_data.prompt,
                "aspect_ratio": input_data.aspect_ratio.value,
                "output_megapixels": input_data.output_megapixels.value,
                "go_fast": input_data.go_fast,
                "output_format": input_data.output_format.value,
                "output_quality": input_data.output_quality,
                "disable_safety_checker": input_data.disable_safety_checker,
            }
            if input_data.seed is not None:
                request_data["seed"] = input_data.seed
            if input_data.images:
                image_urls = []
                for img in input_data.images:
                    if img.uri and img.uri.startswith("http"):
                        image_urls.append(img.uri)
                    else:
                        upload_result = upload_file(img.path, logger=self.logger)
                        image_urls.append(upload_result.get("urls", {}).get("get"))
                request_data["images"] = image_urls

            result = await run_prediction(model=self.model, input_data=request_data, use_sync=True, logger=self.logger)
            generation_url = get_generation_url(result)

            image_path = download_image(generation_url, logger=self.logger)

            # Read input image dimensions if provided
            input_metas = []
            if input_data.images:
                from PIL import Image
                for img in input_data.images:
                    with Image.open(img.path) as pil_img:
                        w, h = pil_img.size
                    input_metas.append(ImageMeta(width=w, height=h, count=1))

            # Calculate dimensions from aspect ratio and megapixels
            mp_sizes = {"0.25": 512, "0.5": 724, "1": 1024, "2": 1448, "4": 2048}
            base_size = mp_sizes.get(input_data.output_megapixels.value, 1024)
            aspect_ratios = {
                "1:1": (1, 1), "16:9": (16, 9), "9:16": (9, 16),
                "21:9": (21, 9), "9:21": (9, 21), "3:2": (3, 2), "2:3": (2, 3),
                "4:5": (4, 5), "5:4": (5, 4), "3:4": (3, 4), "4:3": (4, 3),
            }
            ar = input_data.aspect_ratio.value
            if ar == "match_input_image":
                width, height = base_size, base_size  # fallback
            else:
                w_ratio, h_ratio = aspect_ratios.get(ar, (1, 1))
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

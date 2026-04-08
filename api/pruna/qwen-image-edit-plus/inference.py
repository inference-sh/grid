"""
Qwen-Image-Edit-Plus - Image editing with text instructions and pose transfer
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field
from typing import Optional, List, Union
from enum import Enum
import logging

from .pruna_helper import run_prediction, get_generation_url, download_image, upload_file


class AspectRatioEnum(str, Enum):
    match_input = "match_input_image"
    landscape_16_9 = "16:9"
    portrait_9_16 = "9:16"
    square = "1:1"
    landscape_4_3 = "4:3"
    portrait_3_4 = "3:4"


class OutputFormatEnum(str, Enum):
    webp = "webp"
    jpg = "jpg"
    png = "png"


class AppInput(BaseAppInput):
    prompt: str = Field(description="Text description of the desired edit.")
    images: List[File] = Field(description="1-2 images for editing/composition.", min_length=1, max_length=2)
    go_fast: bool = Field(default=True, description="Run faster with optimizations.")
    aspect_ratio: AspectRatioEnum = Field(default=AspectRatioEnum.match_input, description="Output aspect ratio.")
    seed: Optional[int] = Field(default=None, description="Random seed.")
    output_format: OutputFormatEnum = Field(default=OutputFormatEnum.webp, description="Output format.")
    output_quality: int = Field(default=95, ge=0, le=100, description="Quality for jpg/webp.")
    disable_safety_checker: bool = Field(default=False, description="Disable safety checker.")


class AppOutput(BaseAppOutput):
    image: File = Field(description="Edited image file.")


class App(BaseApp):
    async def setup(self, metadata):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.model = "qwen-image-edit-plus"

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        try:
            self.logger.info(f"Editing: {input_data.prompt[:100]}...")
            
            # Upload images
            image_urls = []
            for img in input_data.images:
                if img.uri and img.uri.startswith("http"):
                    image_urls.append(img.uri)
                else:
                    upload_result = upload_file(img.path, logger=self.logger)
                    image_urls.append(upload_result.get("urls", {}).get("get"))

            # API expects single image or array
            image_param = image_urls[0] if len(image_urls) == 1 else image_urls

            request_data = {
                "prompt": input_data.prompt,
                "image": image_param,
                "go_fast": input_data.go_fast,
                "aspect_ratio": input_data.aspect_ratio.value,
                "output_format": input_data.output_format.value,
                "output_quality": input_data.output_quality,
                "disable_safety_checker": input_data.disable_safety_checker,
            }
            if input_data.seed is not None:
                request_data["seed"] = input_data.seed

            result = await run_prediction(model=self.model, input_data=request_data, use_sync=True, logger=self.logger)
            generation_url = get_generation_url(result)

            image_path = download_image(generation_url, logger=self.logger)

            # Read input image dimensions
            from PIL import Image
            input_metas = []
            for img in input_data.images:
                with Image.open(img.path) as pil_img:
                    w, h = pil_img.size
                input_metas.append(ImageMeta(width=w, height=h, count=1))

            # Read output dimensions
            with Image.open(image_path) as pil_img:
                width, height = pil_img.size

            return AppOutput(image=File(path=image_path), output_meta=OutputMeta(inputs=input_metas, outputs=[ImageMeta(width=width, height=height, count=1)]))
        except Exception as e:
            self.logger.error(f"Error: {e}")
            raise RuntimeError(f"Editing failed: {str(e)}")

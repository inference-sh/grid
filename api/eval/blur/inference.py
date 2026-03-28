import logging
import numpy as np
from typing import Literal
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field
from PIL import Image, ImageFilter
import cv2


class RunInput(BaseAppInput):
    image: File = Field(description="Image to degrade")
    blur_radius: int = Field(default=5, ge=0, le=30, description="Gaussian blur radius (0 = no blur)")
    noise_sigma: float = Field(default=0.0, ge=0.0, le=100.0, description="Gaussian noise sigma (0 = no noise)")
    jpeg_quality: int = Field(default=95, ge=1, le=100, description="JPEG compression quality (lower = more artifacts)")


class RunOutput(BaseAppOutput):
    image: File = Field(description="Degraded image")


class App(BaseApp):
    async def setup(self, config):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Blur/degrade app ready")

    async def run(self, input_data: RunInput) -> RunOutput:
        print(f"[blur] Degrading image: blur={input_data.blur_radius}, noise={input_data.noise_sigma}, jpeg_q={input_data.jpeg_quality}")

        img = Image.open(input_data.image.path).convert("RGB")

        # Gaussian blur
        if input_data.blur_radius > 0:
            img = img.filter(ImageFilter.GaussianBlur(radius=input_data.blur_radius))

        # Gaussian noise
        if input_data.noise_sigma > 0:
            arr = np.array(img, dtype=np.float64)
            noise = np.random.normal(0, input_data.noise_sigma, arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)

        # JPEG compression artifacts
        output_path = "/tmp/degraded.jpg"
        img.save(output_path, "JPEG", quality=input_data.jpeg_quality)

        # Re-read to get actual dimensions
        with Image.open(output_path) as out:
            width, height = out.size

        print(f"[blur] Output: {width}x{height}")

        return RunOutput(
            image=File(path=output_path),
            output_meta=OutputMeta(
                outputs=[ImageMeta(width=width, height=height, count=1)]
            ),
        )

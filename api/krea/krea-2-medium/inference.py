import os
import logging
from typing import List, Optional
from enum import Enum

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import BaseModel, Field

from .krea_helper import KreaClient, download_file


class AspectRatioEnum(str, Enum):
    r1_1 = "1:1"
    r4_3 = "4:3"
    r3_2 = "3:2"
    r16_9 = "16:9"
    r2_35_1 = "2.35:1"
    r4_5 = "4:5"
    r2_3 = "2:3"
    r9_16 = "9:16"


class CreativityEnum(str, Enum):
    raw = "raw"
    low = "low"
    medium = "medium"
    high = "high"


class StyleConfig(BaseModel):
    id: str = Field(description="Trained style/LoRA ID.")
    strength: float = Field(default=1.0, ge=-2.0, le=2.0, description="Style strength.")


class StyleReference(BaseModel):
    image: File = Field(description="Style reference image.")
    strength: float = Field(default=0.5, ge=0.0, le=1.0, description="Reference strength.")


class AppInput(BaseAppInput):
    prompt: str = Field(description="Text prompt for image generation.")
    aspect_ratio: AspectRatioEnum = Field(default=AspectRatioEnum.r1_1, description="Output aspect ratio.")
    creativity: CreativityEnum = Field(default=CreativityEnum.low, description="Creativity level.")
    intensity: int = Field(default=0, ge=-100, le=100, description="Intensity control.")
    complexity: int = Field(default=0, ge=-100, le=100, description="Complexity control.")
    movement: int = Field(default=0, ge=-100, le=100, description="Movement control.")
    seed: Optional[int] = Field(default=None, description="Seed for reproducible results.")
    image: Optional[File] = Field(default=None, description="Input image for image-to-image generation.")
    strength: float = Field(default=0.99, ge=0.0, le=1.0, description="Denoising strength for image-to-image.")
    styles: Optional[List[StyleConfig]] = Field(default=None, description="Trained style/LoRA configurations.")
    image_style_references: Optional[List[StyleReference]] = Field(default=None, description="Style reference images (max 10).")


class AppOutput(BaseAppOutput):
    image: File = Field(description="Generated image.")


class App(BaseApp):
    async def setup(self):
        self.logger = logging.getLogger(__name__)
        api_key = os.environ.get("KREA_KEY")
        if not api_key:
            raise RuntimeError("KREA_KEY must be set")
        self.client = KreaClient(api_key=api_key, logger=self.logger)
        self.logger.info("Krea 2 Medium initialized")

    async def on_cancel(self):
        return True

    async def run(self, input_data: AppInput) -> AppOutput:
        self.logger.info(f"Generating image, aspect_ratio={input_data.aspect_ratio.value}")

        payload = {
            "prompt": input_data.prompt,
            "aspect_ratio": input_data.aspect_ratio.value,
            "resolution": "1K",
            "creativity": input_data.creativity.value,
            "intensity": input_data.intensity,
            "complexity": input_data.complexity,
            "movement": input_data.movement,
        }
        if input_data.seed is not None:
            payload["seed"] = input_data.seed
        if input_data.image:
            payload["image_url"] = input_data.image.uri
            payload["strength"] = input_data.strength
        if input_data.styles:
            payload["styles"] = [{"id": s.id, "strength": s.strength} for s in input_data.styles]
        if input_data.image_style_references:
            payload["image_style_references"] = [
                {"url": r.image.uri, "strength": r.strength}
                for r in input_data.image_style_references
            ]

        result = await self.client.generate("/generate/image/krea/krea-2/medium", payload)

        urls = (result.get("result") or {}).get("urls") or result.get("urls") or result.get("images")
        if not urls:
            raise RuntimeError(f"No output URLs in response: {str(result)[:300]}")

        image_url = urls[0]
        self.logger.info(f"Image ready: {image_url[:80]}...")
        image_path = await download_file(image_url, suffix=".png", logger=self.logger)

        from PIL import Image as PILImage
        with PILImage.open(image_path) as img:
            width, height = img.size
        self.logger.info(f"Generated {width}x{height} image")

        input_metas = []
        if input_data.image:
            input_metas.append(ImageMeta())
        if input_data.image_style_references:
            input_metas.extend(ImageMeta() for _ in input_data.image_style_references)

        return AppOutput(
            image=File(path=image_path),
            output_meta=OutputMeta(
                inputs=input_metas,
                outputs=[ImageMeta(width=width, height=height, count=1)],
            ),
        )

    async def unload(self):
        await self.client.close()

"""
P-Image-Ideogram - High-quality text-to-image generation by Pruna, in collaboration with Ideogram.

Strong typography and prompt understanding. A `thinking` level trades speed against
quality, and output is generated at either a 1K or 2K resolution budget.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field
from typing import Optional
from enum import Enum
import logging

from .pruna_helper import run_prediction, get_generation_url, download_result


class ThinkingEnum(str, Enum):
    very_low = "very low"
    low = "low"
    medium = "medium"
    high = "high"


class ImageSizeEnum(str, Enum):
    k1 = "1K"
    k2 = "2K"


class AspectRatioEnum(str, Enum):
    square = "1:1"
    landscape = "16:9"
    portrait = "9:16"
    photo_landscape = "4:3"
    photo_portrait = "3:4"
    classic_landscape = "3:2"
    classic_portrait = "2:3"
    custom = "custom"


class OutputFormatEnum(str, Enum):
    jpg = "jpg"
    png = "png"
    webp = "webp"


class AppInput(BaseAppInput):
    """Input schema for P-Image-Ideogram."""

    prompt: str = Field(
        description="Text description of the image to generate. Handles rendered text and typography well."
    )
    thinking: ThinkingEnum = Field(
        default=ThinkingEnum.high,
        description="Reasoning effort. Higher levels improve quality at the cost of speed and price."
    )
    image_size: ImageSizeEnum = Field(
        default=ImageSizeEnum.k1,
        description="Output resolution budget. Ignored when aspect_ratio is custom."
    )
    aspect_ratio: AspectRatioEnum = Field(
        default=AspectRatioEnum.square,
        description="Aspect ratio for the image. Use custom to set width and height directly."
    )
    width: Optional[int] = Field(
        default=None,
        ge=256,
        le=2560,
        description="Custom width in pixels (256-2560). Only used when aspect_ratio=custom."
    )
    height: Optional[int] = Field(
        default=None,
        ge=256,
        le=2560,
        description="Custom height in pixels (256-2560). Only used when aspect_ratio=custom."
    )
    prompt_upsampling: bool = Field(
        default=True,
        description="Enhance the prompt with an LLM before generation."
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducible generation."
    )
    output_format: OutputFormatEnum = Field(
        default=OutputFormatEnum.jpg,
        description="Output image format."
    )
    output_quality: int = Field(
        default=80,
        ge=0,
        le=100,
        description="Output quality from 0 to 100. Ignored for PNG."
    )


class AppOutput(BaseAppOutput):
    """Output schema for P-Image-Ideogram."""

    image: File = Field(description="Generated image file.")
    seed: Optional[int] = Field(default=None, description="Seed used for generation.")


class App(BaseApp):
    """P-Image-Ideogram text-to-image generation."""

    async def setup(self):
        """Initialize the application."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.model = "p-image-ideogram"
        self.logger.info("P-Image-Ideogram initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        """Generate an image using P-Image-Ideogram."""
        try:
            self.logger.info(
                f"Generating image (thinking={input_data.thinking.value}, "
                f"image_size={input_data.image_size.value}, "
                f"aspect_ratio={input_data.aspect_ratio.value}): {input_data.prompt[:100]}..."
            )

            request_data = {
                "prompt": input_data.prompt,
                "thinking": input_data.thinking.value,
                "aspect_ratio": input_data.aspect_ratio.value,
                "prompt_upsampling": input_data.prompt_upsampling,
                "output_format": input_data.output_format.value,
                "output_quality": input_data.output_quality,
            }

            # image_size is ignored upstream for custom dimensions
            if input_data.aspect_ratio == AspectRatioEnum.custom:
                if input_data.width:
                    request_data["width"] = input_data.width
                if input_data.height:
                    request_data["height"] = input_data.height
            else:
                request_data["image_size"] = input_data.image_size.value

            if input_data.seed is not None:
                request_data["seed"] = input_data.seed

            result = await run_prediction(
                model=self.model,
                input_data=request_data,
                use_sync=True,
                logger=self.logger,
            )

            generation_url = get_generation_url(result)

            image_path = download_result(
                generation_url,
                suffix=f".{input_data.output_format.value}",
                logger=self.logger,
            )

            # Read actual output dimensions
            from PIL import Image
            with Image.open(image_path) as pil_img:
                width, height = pil_img.size

            resolution_mp = round(width * height / 1_000_000, 4)

            # Upstream bills per (thinking level, resolution budget). For custom
            # dimensions the budget isn't sent, so derive the tier from the
            # actual output area — 1K is ~1MP, 2K is ~4MP.
            if input_data.aspect_ratio == AspectRatioEnum.custom:
                size_tier = ImageSizeEnum.k2.value if resolution_mp > 2.0 else ImageSizeEnum.k1.value
            else:
                size_tier = input_data.image_size.value

            output_meta = OutputMeta(
                outputs=[
                    ImageMeta(
                        width=width,
                        height=height,
                        resolution_mp=resolution_mp,
                        count=1,
                        extra={
                            "thinking": input_data.thinking.value,
                            "image_size": size_tier,
                        },
                    )
                ],
            )

            self.logger.info(f"Generated {width}x{height} image ({size_tier} tier)")

            return AppOutput(
                image=File(path=image_path),
                seed=result.get("seed", input_data.seed),
                output_meta=output_meta,
            )

        except Exception as e:
            self.logger.error(f"Error: {e}")
            raise RuntimeError(f"Image generation failed: {str(e)}")

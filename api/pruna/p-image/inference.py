"""
P-Image - Ultra-fast text-to-image generation by Pruna
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field
from typing import Optional, Literal
from enum import Enum
import logging

from .pruna_helper import run_prediction, get_generation_url, download_image


class AspectRatioEnum(str, Enum):
    square = "1:1"
    landscape = "16:9"
    portrait = "9:16"
    photo_landscape = "4:3"
    photo_portrait = "3:4"
    classic_landscape = "3:2"
    classic_portrait = "2:3"
    custom = "custom"


class AppInput(BaseAppInput):
    """Input schema for P-Image."""

    prompt: str = Field(
        description="Text description of the image to generate. Model auto-enhances prompts."
    )
    aspect_ratio: AspectRatioEnum = Field(
        default=AspectRatioEnum.landscape,
        description="Aspect ratio for the image."
    )
    width: Optional[int] = Field(
        default=None,
        ge=256,
        le=1440,
        description="Custom width in pixels (256-1440, multiple of 16). Only used when aspect_ratio=custom."
    )
    height: Optional[int] = Field(
        default=None,
        ge=256,
        le=1440,
        description="Custom height in pixels (256-1440, multiple of 16). Only used when aspect_ratio=custom."
    )
    lora_weights: Optional[str] = Field(
        default=None,
        description="HuggingFace LoRA URL: huggingface.co/owner/repo[/file.safetensors]"
    )
    lora_scale: Optional[float] = Field(
        default=None,
        ge=-1,
        le=3,
        description="LoRA strength (-1 to 3). Default 0.5 works well for most."
    )
    hf_api_token: Optional[str] = Field(
        default=None,
        description="HuggingFace API token for private LoRAs."
    )
    prompt_upsampling: bool = Field(
        default=False,
        description="Enhance prompt with LLM for better results."
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducible generation."
    )
    disable_safety_checker: bool = Field(
        default=False,
        description="Disable safety checker for generated images."
    )


class AppOutput(BaseAppOutput):
    """Output schema for P-Image."""

    image: File = Field(description="Generated image file.")
    seed: Optional[int] = Field(default=None, description="Seed used for generation.")


class App(BaseApp):
    """P-Image text-to-image generation."""

    async def setup(self):
        """Initialize the application."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.model = "p-image"
        self.logger.info("P-Image initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        """Generate an image using P-Image."""
        try:
            self.logger.info(f"Generating image: {input_data.prompt[:100]}...")

            # Build request
            request_data = {
                "prompt": input_data.prompt,
                "aspect_ratio": input_data.aspect_ratio.value,
            }

            # Add optional parameters
            if input_data.aspect_ratio == AspectRatioEnum.custom:
                if input_data.width:
                    request_data["width"] = input_data.width
                if input_data.height:
                    request_data["height"] = input_data.height

            if input_data.lora_weights:
                request_data["lora_weights"] = input_data.lora_weights
            if input_data.lora_scale is not None:
                request_data["lora_scale"] = input_data.lora_scale
            if input_data.hf_api_token:
                request_data["hf_api_token"] = input_data.hf_api_token
            if input_data.prompt_upsampling:
                request_data["prompt_upsampling"] = input_data.prompt_upsampling
            if input_data.seed is not None:
                request_data["seed"] = input_data.seed
            if input_data.disable_safety_checker:
                request_data["disable_safety_checker"] = input_data.disable_safety_checker

            # Run prediction (sync mode for fast image generation)
            result = await run_prediction(
                model=self.model,
                input_data=request_data,
                use_sync=True,
                logger=self.logger,
            )

            generation_url = get_generation_url(result)

            image_path = download_image(generation_url, logger=self.logger)

            # Read actual output dimensions
            from PIL import Image
            with Image.open(image_path) as pil_img:
                width, height = pil_img.size

            output_meta = OutputMeta(
                outputs=[ImageMeta(width=width, height=height, count=1)],
            )

            self.logger.info("Image generated successfully")

            return AppOutput(
                image=File(path=image_path),
                seed=result.get("seed"),
                output_meta=output_meta,
            )

        except Exception as e:
            self.logger.error(f"Error: {e}")
            raise RuntimeError(f"Image generation failed: {str(e)}")

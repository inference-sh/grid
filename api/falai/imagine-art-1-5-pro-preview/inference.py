"""
ImagineArt 1.5 Pro Preview - Text-to-Image Generation

An advanced text-to-image model that creates ultra-high-fidelity 4K visuals
with lifelike realism, refined aesthetics, and powerful creative output
suited for professional use.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field
from typing import Optional
from enum import Enum
import logging

from .fal_helper import setup_fal_client, run_fal_model, download_image


class AspectRatio(str, Enum):
    """Supported aspect ratios for image generation."""
    SQUARE = "1:1"
    LANDSCAPE_16_9 = "16:9"
    PORTRAIT_9_16 = "9:16"
    LANDSCAPE_4_3 = "4:3"
    PORTRAIT_3_4 = "3:4"
    ULTRAWIDE_3_1 = "3:1"
    ULTRATALL_1_3 = "1:3"
    LANDSCAPE_3_2 = "3:2"
    PORTRAIT_2_3 = "2:3"


# Map aspect ratios to approximate dimensions for metadata
ASPECT_RATIO_DIMENSIONS = {
    "1:1": (1024, 1024),
    "16:9": (1820, 1024),
    "9:16": (1024, 1820),
    "4:3": (1365, 1024),
    "3:4": (1024, 1365),
    "3:1": (1820, 607),
    "1:3": (607, 1820),
    "3:2": (1536, 1024),
    "2:3": (1024, 1536),
}


class AppInput(BaseAppInput):
    """Input parameters for ImagineArt 1.5 Pro Preview."""
    prompt: str = Field(
        description="Text prompt describing the desired image. Be descriptive for best results."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.SQUARE,
        description="Image aspect ratio. Choose based on your intended use case."
    )
    seed: Optional[int] = Field(
        default=None,
        description="Seed for reproducible generation. Leave empty for random results."
    )


class AppOutput(BaseAppOutput):
    """Output from ImagineArt 1.5 Pro Preview."""
    image: File = Field(description="The generated image.")
    seed: Optional[int] = Field(default=None, description="The seed used for generation.")


class App(BaseApp):
    """ImagineArt 1.5 Pro Preview text-to-image generation app."""

    async def setup(self, metadata):
        """Initialize the application."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Suppress noisy httpx polling logs
        logging.getLogger("httpx").setLevel(logging.WARNING)
        
        self.model_id = "imagineart/imagineart-1.5-pro-preview/text-to-image"
        self.logger.info(f"ImagineArt 1.5 Pro Preview initialized with model: {self.model_id}")

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Generate an image using ImagineArt 1.5 Pro Preview via fal.ai."""
        try:
            # Set up API key from environment
            setup_fal_client()
            
            self.logger.info(f"Generating image with prompt: '{input_data.prompt[:100]}...'")
            self.logger.info(f"Aspect ratio: {input_data.aspect_ratio.value}")
            
            # Build request
            request_data = {
                "prompt": input_data.prompt,
                "aspect_ratio": input_data.aspect_ratio.value,
            }
            if input_data.seed is not None:
                request_data["seed"] = input_data.seed
            
            # Run model inference using helper
            result = run_fal_model(
                self.model_id,
                request_data,
                self.logger
            )
            
            # Extract the generated image
            images = result.get("images", [])
            if not images:
                raise RuntimeError("No images returned from the model")
            
            image_data = images[0]
            image_url = image_data["url"]
            
            # Download image using helper
            image_path = download_image(image_url, self.logger)
            
            # Get dimensions from aspect ratio for metadata
            width, height = ASPECT_RATIO_DIMENSIONS.get(
                input_data.aspect_ratio.value, 
                (1024, 1024)
            )
            
            # Use actual dimensions from response if available (check for truthy values)
            if image_data.get("width"):
                width = image_data["width"]
            if image_data.get("height"):
                height = image_data["height"]
            
            # Build output metadata for pricing
            output_meta = OutputMeta(
                outputs=[
                    ImageMeta(
                        width=width,
                        height=height,
                        count=1
                    )
                ]
            )
            
            # Get seed from response if available
            result_seed = result.get("seed")
            
            return AppOutput(
                image=File(path=image_path),
                seed=result_seed,
                output_meta=output_meta
            )
            
        except Exception as e:
            self.logger.error(f"Error during image generation: {e}")
            raise RuntimeError(f"Image generation failed: {str(e)}")

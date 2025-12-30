"""
Flux 2 Turbo - Text-to-Image & Image-to-Image Generation

Text-to-image generation with FLUX.2 [dev] from Black Forest Labs. 
Enhanced realism, crisper text generation, and native editing capabilities—all at turbo speed.

When reference_images are provided, uses the edit endpoint for image-to-image transformation.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field
from typing import Optional, List
from enum import Enum
import fal_client
import tempfile
import os
import logging
import requests

# Suppress noisy httpx polling logs
logging.getLogger("httpx").setLevel(logging.WARNING)


class OutputFormat(str, Enum):
    """Output image format."""
    PNG = "png"
    JPEG = "jpeg"
    WEBP = "webp"


class AppInput(BaseAppInput):
    """Input schema for Flux 2 Turbo image generation."""
    
    prompt: str = Field(
        description="The text prompt to generate an image from.",
        examples=["A realistic photograph of a vintage typewriter with a sheet of paper inserted that says 'Chapter One: The Journey Begins,' sunlight falling across the desk."]
    )
    height: int = Field(
        default=1024, 
        ge=512, 
        le=2048,
        description="The height in pixels of the generated image."
    )
    width: int = Field(
        default=1024, 
        ge=512, 
        le=2048,
        description="The width in pixels of the generated image."
    )
    reference_images: Optional[List[File]] = Field(
        default=None, 
        description="List of reference images to use for editing. When provided, uses image-to-image mode. Maximum 4 images."
    )
    guidance_scale: float = Field(
        default=2.5, 
        ge=0, 
        le=20,
        description="Guidance scale - how closely the model follows the prompt. Lower values give more creative freedom."
    )
    seed: Optional[int] = Field(
        default=None, 
        description="The seed for random generation. If not provided, a random seed will be used."
    )
    num_images: int = Field(
        default=1, 
        ge=1, 
        le=4,
        description="The number of images to generate."
    )
    enable_prompt_expansion: bool = Field(
        default=False, 
        description="If set to true, the prompt will be expanded for better results."
    )
    enable_safety_checker: bool = Field(
        default=True, 
        description="If set to true, the safety checker will be enabled."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, 
        description="The format of the generated image."
    )


class AppOutput(BaseAppOutput):
    """Output schema for Flux 2 Turbo image generation."""
    
    images: List[File] = Field(description="The generated image(s).")
    seed: int = Field(description="The seed used for generation.")
    prompt: str = Field(description="The prompt used for generating the image.")
    has_nsfw_concepts: List[bool] = Field(description="Whether each generated image contains NSFW concepts.")


class App(BaseApp):
    """Flux 2 Turbo app for text-to-image and image-to-image generation via fal.ai."""
    
    async def setup(self, metadata):
        """Initialize the application."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Model endpoints
        self.text_to_image_model = "fal-ai/flux-2/turbo"
        self.edit_model = "fal-ai/flux-2/turbo/edit"
        
        self.logger.info("Flux 2 Turbo app initialized successfully")

    def _setup_fal_client(self) -> str:
        """Configure fal.ai client with API key."""
        api_key = os.environ.get("FAL_KEY")
        if not api_key:
            raise RuntimeError("FAL_KEY environment variable is required for fal.ai API access.")
        fal_client.api_key = api_key
        return api_key

    def _build_request(self, input_data: AppInput, image_urls: Optional[List[str]] = None) -> dict:
        """Build the request payload for fal.ai."""
        arguments = {
            "prompt": input_data.prompt,
            "image_size": {
                "width": input_data.width,
                "height": input_data.height
            },
            "guidance_scale": input_data.guidance_scale,
            "num_images": input_data.num_images,
            "enable_prompt_expansion": input_data.enable_prompt_expansion,
            "enable_safety_checker": input_data.enable_safety_checker,
            "output_format": input_data.output_format.value,
        }
        
        if input_data.seed is not None:
            arguments["seed"] = input_data.seed
            
        if image_urls:
            arguments["image_urls"] = image_urls
            
        return arguments

    def _download_image(self, url: str, output_format: str) -> str:
        """Download an image from URL to a temporary file."""
        suffix = f".{output_format}"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
            image_path = tmp_file.name
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(image_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return image_path

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Generate images using Flux 2 Turbo model via fal.ai."""
        try:
            self._setup_fal_client()
            
            # Determine which model to use based on reference images
            if input_data.reference_images:
                model_id = self.edit_model
                # Use URI for remote access by fal.ai
                image_urls = [img.uri for img in input_data.reference_images[:4]]  # Max 4 images
                self.logger.info(f"Using edit mode with {len(image_urls)} reference image(s)")
            else:
                model_id = self.text_to_image_model
                image_urls = None
                self.logger.info("Using text-to-image mode")
            
            # Build request
            arguments = self._build_request(input_data, image_urls)
            
            self.logger.info(f"Generating image for prompt: '{input_data.prompt[:100]}...'")
            self.logger.info(f"Settings: {input_data.width}x{input_data.height}, guidance={input_data.guidance_scale}")
            
            # Define progress callback
            def on_queue_update(update):
                if isinstance(update, fal_client.InProgress):
                    for log in update.logs:
                        self.logger.info(f"fal.ai: {log['message']}")
            
            # Run model inference
            result = fal_client.subscribe(
                model_id,
                arguments=arguments,
                with_logs=True,
                on_queue_update=on_queue_update,
            )
            
            self.logger.info("Image generation completed successfully")
            
            # Download all generated images
            output_images = []
            for i, img_data in enumerate(result["images"]):
                image_url = img_data["url"]
                self.logger.info(f"Downloading image {i + 1}/{len(result['images'])}...")
                image_path = self._download_image(image_url, input_data.output_format.value)
                output_images.append(File(path=image_path))
            
            self.logger.info(f"Generated {len(output_images)} image(s)")
            
            # Build output metadata for pricing
            output_meta = OutputMeta(
                outputs=[
                    ImageMeta(
                        width=input_data.width,
                        height=input_data.height,
                        count=len(output_images)
                    )
                ]
            )
            
            return AppOutput(
                images=output_images,
                seed=result["seed"],
                prompt=result.get("prompt", input_data.prompt),
                has_nsfw_concepts=result.get("has_nsfw_concepts", [False] * len(output_images)),
                output_meta=output_meta
            )
            
        except Exception as e:
            self.logger.error(f"Error during image generation: {e}")
            raise RuntimeError(f"Image generation failed: {str(e)}")

"""
FLUX.1 [dev] LoRA - Text-to-Image & Image-to-Image Generation

Super fast endpoint for the FLUX.1 [dev] model with LoRA support, enabling rapid
and high-quality image generation using pre-trained LoRA adaptations for personalization,
specific styles, brand identities, and product-specific outputs.

When an image is provided, uses the image-to-image endpoint for guided transformation.
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


class LoRAConfig(BaseAppInput):
    """Configuration for a LoRA adapter."""
    path: str = Field(
        description="URL or the path to the LoRA weights (.safetensors). Supports HuggingFace repos, Civitai URLs, or direct URLs."
    )
    scale: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="The scale of the LoRA weight. This is used to scale the LoRA weight before merging it with the base model."
    )


class AppInput(BaseAppInput):
    """Input schema for FLUX.1 [dev] LoRA image generation."""

    prompt: str = Field(
        description="The prompt to generate an image from.",
        examples=["Extreme close-up of a single tiger eye, direct frontal view. Detailed iris and pupil. Sharp focus on eye texture and color."]
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
    image: Optional[File] = Field(
        default=None,
        description="Optional input image for image-to-image mode. When provided, the model will transform this image based on the prompt."
    )
    strength: float = Field(
        default=0.85,
        ge=0.01,
        le=1.0,
        description="How much to transform the input image (image-to-image only). 1.0 = full remake, 0.0 = preserve original."
    )
    loras: Optional[List[LoRAConfig]] = Field(
        default=None,
        description="The LoRAs to use for the image generation. You can use any number of LoRAs and they will be merged together to generate the final image."
    )
    num_inference_steps: int = Field(
        default=28,
        ge=1,
        le=50,
        description="The number of inference steps to perform."
    )
    guidance_scale: float = Field(
        default=3.5,
        ge=0,
        le=35,
        description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    seed: Optional[int] = Field(
        default=None,
        description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )
    num_images: int = Field(
        default=1,
        ge=1,
        le=4,
        description="The number of images to generate."
    )
    enable_safety_checker: bool = Field(
        default=True,
        description="If set to true, the safety checker will be enabled."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG,
        description="The format of the generated image."
    )


class AppOutput(BaseAppOutput):
    """Output schema for FLUX.1 [dev] LoRA image generation."""

    images: List[File] = Field(description="The generated image(s).")
    seed: int = Field(description="The seed used for generation.")
    prompt: str = Field(description="The prompt used for generating the image.")
    has_nsfw_concepts: List[bool] = Field(description="Whether each generated image contains NSFW concepts.")


class App(BaseApp):
    """FLUX.1 [dev] LoRA app for text-to-image and image-to-image generation via fal.ai."""

    async def setup(self, metadata):
        """Initialize the application."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Model endpoints
        self.t2i_endpoint = "fal-ai/flux-lora"
        self.i2i_endpoint = "fal-ai/flux-lora/image-to-image"

        self.logger.info("FLUX.1 [dev] LoRA app initialized successfully")

    def _setup_fal_client(self) -> str:
        """Configure fal.ai client with API key."""
        api_key = os.environ.get("FAL_KEY")
        if not api_key:
            raise RuntimeError("FAL_KEY environment variable is required for fal.ai API access.")
        fal_client.api_key = api_key
        return api_key

    def _build_request(self, input_data: AppInput, image_url: Optional[str] = None) -> dict:
        """Build the request payload for fal.ai."""
        arguments = {
            "prompt": input_data.prompt,
            "image_size": {
                "width": input_data.width,
                "height": input_data.height
            },
            "num_inference_steps": input_data.num_inference_steps,
            "guidance_scale": input_data.guidance_scale,
            "num_images": input_data.num_images,
            "enable_safety_checker": input_data.enable_safety_checker,
            "output_format": input_data.output_format.value,
        }

        # Add seed if provided
        if input_data.seed is not None:
            arguments["seed"] = input_data.seed

        # Add LoRA configurations if provided
        if input_data.loras:
            lora_list = []
            for lora in input_data.loras:
                lora_list.append({
                    "path": lora.path,
                    "scale": lora.scale
                })
            arguments["loras"] = lora_list

        # Add image URL and strength for image-to-image mode
        if image_url:
            arguments["image_url"] = image_url
            arguments["strength"] = input_data.strength

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
        """Generate images using FLUX.1 [dev] LoRA model via fal.ai."""
        try:
            self._setup_fal_client()

            # Determine which endpoint to use based on image input
            if input_data.image:
                model_id = self.i2i_endpoint
                image_url = input_data.image.uri
                self.logger.info(f"Using image-to-image mode with strength={input_data.strength}")
            else:
                model_id = self.t2i_endpoint
                image_url = None
                self.logger.info("Using text-to-image mode")

            # Log LoRA info if provided
            if input_data.loras:
                self.logger.info(f"Applying {len(input_data.loras)} LoRA(s)")

            # Build request
            arguments = self._build_request(input_data, image_url)

            self.logger.info(f"Generating image for prompt: '{input_data.prompt[:100]}...'")
            self.logger.info(f"Settings: {input_data.width}x{input_data.height}, steps={input_data.num_inference_steps}, guidance={input_data.guidance_scale}")

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

            # Build output metadata for pricing (megapixel-based)
            output_resolution_mp = (input_data.width * input_data.height) / 1_000_000

            # Track input image for I2I mode
            inputs_meta = []
            if input_data.image:
                inputs_meta.append(
                    ImageMeta(
                        resolution_mp=1.0,  # Input images typically resized
                        count=1
                    )
                )

            output_meta = OutputMeta(
                inputs=inputs_meta if inputs_meta else None,
                outputs=[
                    ImageMeta(
                        width=input_data.width,
                        height=input_data.height,
                        resolution_mp=output_resolution_mp,
                        count=len(output_images)
                    )
                ]
            )

            return AppOutput(
                images=output_images,
                has_nsfw_concepts=result.get("has_nsfw_concepts", [False] * len(output_images)),
                output_meta=output_meta
            )

        except Exception as e:
            self.logger.error(f"Error during image generation: {e}")
            raise RuntimeError(f"Image generation failed: {str(e)}")

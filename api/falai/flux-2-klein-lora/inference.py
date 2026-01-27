"""
FLUX.2 [klein] LoRA - Text-to-Image & Image-to-Image Generation

Text-to-image generation with LoRA support for FLUX.2 [klein] from Black Forest Labs.
Custom style adaptation and fine-tuned model variations. Available in 4B and 9B
parameter sizes.

When images are provided, uses the edit endpoint for image-to-image transformation.
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


class ModelSize(str, Enum):
    """Model parameter size."""
    KLEIN_4B = "4b"
    KLEIN_9B = "9b"


class AccelerationLevel(str, Enum):
    """Acceleration level for image generation."""
    NONE = "none"
    REGULAR = "regular"
    HIGH = "high"


class OutputFormat(str, Enum):
    """Output image format."""
    PNG = "png"
    JPEG = "jpeg"
    WEBP = "webp"


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
    """Input schema for FLUX.2 [klein] LoRA image generation."""

    prompt: str = Field(
        description="The prompt to generate an image from.",
        examples=["A serene Japanese garden with cherry blossoms, koi pond, and traditional wooden bridge at golden hour"]
    )
    negative_prompt: str = Field(
        default="",
        description="Negative prompt for classifier-free guidance. Describes what to avoid in the image."
    )
    model_size: ModelSize = Field(
        default=ModelSize.KLEIN_4B,
        description="Model parameter size. 4B is faster, 9B offers higher quality."
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
    images: Optional[List[File]] = Field(
        default=None,
        description="List of images for editing. When provided, uses image-to-image mode. Maximum 4 images."
    )
    loras: Optional[List[LoRAConfig]] = Field(
        default=None,
        description="List of LoRA weights to apply (maximum 3)."
    )
    num_inference_steps: int = Field(
        default=28,
        ge=4,
        le=50,
        description="The number of inference steps to perform."
    )
    guidance_scale: float = Field(
        default=5.0,
        ge=0,
        le=20,
        description="Guidance scale for classifier-free guidance."
    )
    seed: Optional[int] = Field(
        default=None,
        description="The seed to use for the generation. If not provided, a random seed will be used."
    )
    num_images: int = Field(
        default=1,
        ge=1,
        le=4,
        description="The number of images to generate."
    )
    acceleration: AccelerationLevel = Field(
        default=AccelerationLevel.REGULAR,
        description="The acceleration level to use for image generation."
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
    """Output schema for FLUX.2 [klein] LoRA image generation."""

    images: List[File] = Field(description="The generated image(s).")
    seed: int = Field(description="The seed used for generation.")
    prompt: str = Field(description="The prompt used for generating the image.")
    has_nsfw_concepts: List[bool] = Field(description="Whether each generated image contains NSFW concepts.")


class App(BaseApp):
    """FLUX.2 [klein] LoRA app for text-to-image and image-to-image generation via fal.ai."""

    async def setup(self, metadata):
        """Initialize the application."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Model endpoint templates
        self.model_endpoints = {
            ModelSize.KLEIN_4B: {
                "t2i": "fal-ai/flux-2/klein/4b/base/lora",
                "edit": "fal-ai/flux-2/klein/4b/base/edit/lora",
            },
            ModelSize.KLEIN_9B: {
                "t2i": "fal-ai/flux-2/klein/9b/base/lora",
                "edit": "fal-ai/flux-2/klein/9b/base/edit/lora",
            }
        }

        self.logger.info("FLUX.2 [klein] LoRA app initialized successfully")

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
            "num_inference_steps": input_data.num_inference_steps,
            "guidance_scale": input_data.guidance_scale,
            "num_images": input_data.num_images,
            "acceleration": input_data.acceleration.value,
            "enable_safety_checker": input_data.enable_safety_checker,
            "output_format": input_data.output_format.value,
        }

        # Add negative prompt
        arguments["negative_prompt"] = input_data.negative_prompt

        # Add seed if provided
        if input_data.seed is not None:
            arguments["seed"] = input_data.seed

        # Add LoRA configurations if provided
        if input_data.loras:
            lora_list = []
            for lora in input_data.loras[:3]:  # Max 3 LoRAs
                lora_list.append({
                    "path": lora.path,
                    "scale": lora.scale
                })
            arguments["loras"] = lora_list

        # Add image URLs for edit mode
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
        """Generate images using FLUX.2 [klein] LoRA model via fal.ai."""
        try:
            self._setup_fal_client()

            # Determine which model endpoint to use based on images and model size
            endpoints = self.model_endpoints[input_data.model_size]

            if input_data.images:
                model_id = endpoints["edit"]
                # Use URI for remote access by fal.ai
                image_urls = [img.uri for img in input_data.images[:4]]  # Max 4 images
                self.logger.info(f"Using edit mode ({input_data.model_size.value}) with {len(image_urls)} image(s)")
            else:
                model_id = endpoints["t2i"]
                image_urls = None
                self.logger.info(f"Using text-to-image mode ({input_data.model_size.value})")

            # Log LoRA info if provided
            if input_data.loras:
                self.logger.info(f"Applying {len(input_data.loras)} LoRA(s)")

            # Build request
            arguments = self._build_request(input_data, image_urls)

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

            # Build output metadata for pricing
            # Pricing is per megapixel: 4B=$0.016/MP, 9B=$0.02/MP
            # Edit mode: charges for both input (resized to 1MP) and output
            output_resolution_mp = (input_data.width * input_data.height) / 1_000_000

            # Track input images for edit mode (each resized to 1MP by fal.ai)
            inputs_meta = []
            if image_urls:
                inputs_meta.append(
                    ImageMeta(
                        resolution_mp=1.0,  # Input images resized to 1MP
                        count=len(image_urls),
                        extra={"model_size": input_data.model_size.value}
                    )
                )

            output_meta = OutputMeta(
                inputs=inputs_meta if inputs_meta else None,
                outputs=[
                    ImageMeta(
                        width=input_data.width,
                        height=input_data.height,
                        resolution_mp=output_resolution_mp,
                        count=len(output_images),
                        extra={"model_size": input_data.model_size.value}
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

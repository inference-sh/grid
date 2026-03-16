"""
Nano Banana 2

Google's state-of-the-art fast image generation and editing model.
Supports both text-to-image and image editing modes.
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


class AspectRatioEnum(str, Enum):
    """Aspect ratio options for generated images."""
    auto = "auto"
    aspect_21_9 = "21:9"
    aspect_16_9 = "16:9"
    aspect_3_2 = "3:2"
    aspect_4_3 = "4:3"
    aspect_5_4 = "5:4"
    aspect_1_1 = "1:1"
    aspect_4_5 = "4:5"
    aspect_3_4 = "3:4"
    aspect_2_3 = "2:3"
    aspect_9_16 = "9:16"
    aspect_4_1 = "4:1"
    aspect_1_4 = "1:4"
    aspect_8_1 = "8:1"
    aspect_1_8 = "1:8"


class OutputFormatEnum(str, Enum):
    """Output format options."""
    jpeg = "jpeg"
    png = "png"
    webp = "webp"


class ResolutionEnum(str, Enum):
    """Resolution options for generated images."""
    half_k = "0.5K"
    one_k = "1K"
    two_k = "2K"
    four_k = "4K"


class SafetyToleranceEnum(str, Enum):
    """Safety tolerance levels (1=most strict, 6=least strict)."""
    level_1 = "1"
    level_2 = "2"
    level_3 = "3"
    level_4 = "4"
    level_5 = "5"
    level_6 = "6"


class ThinkingLevelEnum(str, Enum):
    """Thinking level for model reasoning."""
    minimal = "minimal"
    high = "high"


class AppInput(BaseAppInput):
    prompt: str = Field(
        description="The text prompt to generate or edit an image. Describe what you want to create or change.",
        min_length=3,
        max_length=50000
    )
    images: Optional[List[File]] = Field(
        default=None,
        description="Optional input images for editing. If provided, enables image editing mode."
    )
    num_images: int = Field(
        default=1,
        ge=1,
        le=4,
        description="Number of images to generate."
    )
    aspect_ratio: AspectRatioEnum = Field(
        default=AspectRatioEnum.auto,
        description="Aspect ratio of the generated image. Use 'auto' to let the model decide based on the prompt."
    )
    resolution: ResolutionEnum = Field(
        default=ResolutionEnum.one_k,
        description="Resolution of the generated image (0.5K, 1K, 2K, or 4K)."
    )
    output_format: OutputFormatEnum = Field(
        default=OutputFormatEnum.png,
        description="Output format for the generated images."
    )
    safety_tolerance: SafetyToleranceEnum = Field(
        default=SafetyToleranceEnum.level_4,
        description="Safety tolerance level. 1 is most strict, 6 is least strict."
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility."
    )
    enable_web_search: bool = Field(
        default=False,
        description="Enable web search to use latest information for generation."
    )
    thinking_level: Optional[ThinkingLevelEnum] = Field(
        default=None,
        description="Enable model thinking with given level ('minimal' or 'high'). Omit to disable."
    )


class AppOutput(BaseAppOutput):
    images: List[File] = Field(description="The generated images")
    description: str = Field(description="Text description from the model")


class App(BaseApp):
    async def setup(self, metadata):
        """Initialize configuration."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Model endpoints
        self.text_to_image_model = "fal-ai/nano-banana-2"
        self.edit_model = "fal-ai/nano-banana-2/edit"

        self.logger.info("Nano Banana 2 initialized successfully")

    def _prepare_request(self, input_data: AppInput) -> dict:
        """Prepare the request payload for fal.ai."""
        request = {
            "prompt": input_data.prompt,
            "num_images": input_data.num_images,
            "aspect_ratio": input_data.aspect_ratio.value,
            "resolution": input_data.resolution.value,
            "output_format": input_data.output_format.value,
            "safety_tolerance": input_data.safety_tolerance.value,
            "enable_web_search": input_data.enable_web_search,
            "limit_generations": True,
        }

        if input_data.seed is not None:
            request["seed"] = input_data.seed

        if input_data.thinking_level is not None:
            request["thinking_level"] = input_data.thinking_level.value

        # Add image URLs for edit mode
        if input_data.images:
            request["image_urls"] = [img.uri for img in input_data.images]

        return request

    def _get_model_id(self, input_data: AppInput) -> str:
        """Select model endpoint based on input."""
        if input_data.images:
            return self.edit_model
        return self.text_to_image_model

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Generate or edit images using Nano Banana 2."""
        try:
            # Validate input images if provided
            if input_data.images:
                for i, image in enumerate(input_data.images):
                    if not image.exists():
                        raise RuntimeError(f"Input image {i+1} does not exist at path: {image.path}")

            # Set up API key
            api_key = os.environ.get("FAL_KEY")
            if not api_key:
                raise RuntimeError("FAL_KEY environment variable is required for model access.")
            fal_client.api_key = api_key

            # Determine mode and model
            model_id = self._get_model_id(input_data)
            mode = "edit" if input_data.images else "text-to-image"
            self.logger.info(f"Using {mode} mode with model: {model_id}")
            self.logger.info(f"Prompt: {input_data.prompt[:100]}...")

            # Prepare request
            request_data = self._prepare_request(input_data)

            # Progress callback
            def on_queue_update(update):
                if isinstance(update, fal_client.InProgress):
                    for log in update.logs:
                        self.logger.info(f"Model: {log['message']}")

            # Run inference
            result = fal_client.subscribe(
                model_id,
                arguments=request_data,
                with_logs=True,
                on_queue_update=on_queue_update,
            )

            self.logger.info("Generation completed successfully")

            # Process output images
            output_images = []
            file_extension = f".{input_data.output_format.value}"

            for i, image_data in enumerate(result["images"]):
                self.logger.info(f"Processing image {i+1}...")

                with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as tmp_file:
                    image_path = tmp_file.name

                # Download image
                image_url = image_data["url"]
                response = requests.get(image_url, stream=True)
                response.raise_for_status()

                with open(image_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                output_images.append(File(path=image_path))

            self.logger.info(f"Generated {len(output_images)} image(s)")

            # Output metadata for pricing
            output_meta = OutputMeta(
                outputs=[
                    ImageMeta(count=len(output_images))
                ]
            )

            return AppOutput(
                images=output_images,
                description=result.get("description", ""),
                output_meta=output_meta
            )

        except Exception as e:
            self.logger.error(f"Error during generation: {e}")
            raise RuntimeError(f"Generation failed: {str(e)}")

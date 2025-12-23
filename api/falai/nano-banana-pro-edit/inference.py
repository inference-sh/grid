from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from typing import Optional, List
from enum import Enum
import fal_client
import tempfile
import os
import logging
import requests

class OutputFormatEnum(str, Enum):
    """Output format options."""
    png = "png"
    jpeg = "jpeg"
    webp = "webp"

class AspectRatioEnum(str, Enum):
    """Aspect ratio options."""
    auto = "auto"
    ratio_21_9 = "21:9"
    ratio_16_9 = "16:9"
    ratio_3_2 = "3:2"
    ratio_4_3 = "4:3"
    ratio_5_4 = "5:4"
    ratio_1_1 = "1:1"
    ratio_4_5 = "4:5"
    ratio_3_4 = "3:4"
    ratio_2_3 = "2:3"
    ratio_9_16 = "9:16"
    ratio_9_21 = "9:21"

class ResolutionEnum(str, Enum):
    """Resolution options."""
    res_1k = "1K"
    res_2k = "2K"
    res_4k = "4K"

class AppInput(BaseAppInput):
    prompt: str = Field(
        description="Editing instruction for the image. Example: 'make a photo of the man driving the car down the california coastline'"
    )
    images: List[File] = Field(
        description="The reference image to edit. Supported formats: PNG, JPEG, WebP, AVIF, and HEIF."
    )
    aspect_ratio: AspectRatioEnum = Field(
        default=AspectRatioEnum.auto,
        description="Aspect ratio for the output image. Default: auto"
    )
    resolution: ResolutionEnum = Field(
        default=ResolutionEnum.res_1k,
        description="Output resolution. Default: 1K"
    )

class AppOutput(BaseAppOutput):
    images: List[File] = Field(description="The edited/generated images")
    description: str = Field(default="", description="Text description of the generated images")

class App(BaseApp):
    async def setup(self, metadata):
        """Initialize model and configuration."""
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Store metadata for later use
        self.metadata = metadata

        # Model endpoint - Gemini 3 Pro Image Preview (Nano Banana Pro)
        self.model_id = "fal-ai/gemini-3-pro-image-preview/edit"

        self.logger.info("Gemini 3 Pro Image Editor (Nano Banana Pro) initialized successfully")

    def _prepare_model_request(self, input_data: AppInput) -> dict:
        """Prepare the request payload for model inference."""
        # Upload image file to get URL
        # Prepare request data - Gemini uses image_urls (array) instead of image_url
        request_data = {
            "prompt": input_data.prompt,
            "image_urls": [image.uri for image in input_data.images],  # Array of image URLs
            "num_images": 1,
            "aspect_ratio": input_data.aspect_ratio.value,
            "output_format": "png",
            "resolution": input_data.resolution.value,
        }

        return request_data

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Edit image using Gemini 3 Pro Image Preview model."""
        try:
            # Set up API key from environment
            api_key = os.environ.get("FAL_KEY")
            if not api_key:
                raise RuntimeError(
                    "FAL_KEY environment variable is required for model access."
                )

            # Configure client with API key
            fal_client.api_key = api_key

            self.logger.info(f"Starting image editing with Gemini 3 Pro")
            self.logger.info(f"Prompt: {input_data.prompt[:100]}...")
            self.logger.info(f"Generating 1 image at {input_data.resolution.value} resolution")

            # Prepare request data for model
            request_data = self._prepare_model_request(input_data)

            self.logger.info("Initializing model inference...")

            # Define progress callback
            def on_queue_update(update):
                if isinstance(update, fal_client.InProgress):
                    for log in update.logs:
                        self.logger.info(f"Model: {log['message']}")

            # Run model inference with progress logging
            result = fal_client.subscribe(
                self.model_id,
                arguments=request_data,
                with_logs=True,
                on_queue_update=on_queue_update,
            )

            self.logger.info("Image editing completed successfully")

            # Extract description from result
            description = result.get("description", "")
            if description:
                self.logger.info(f"Model description: {description}")

            # Process the generated images
            output_images = []
            for i, image_data in enumerate(result["images"]):
                self.logger.info(f"Processing generated image {i+1}/{len(result['images'])}...")

                # Create temporary file for the image
                file_extension = ".png"
                with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as tmp_file:
                    image_path = tmp_file.name

                # Download image content
                # Check if URL is a data URI
                image_url = image_data.get("url", "")
                if "data:" in image_url:
                    # Handle data URI format
                    import base64
                    data_uri = image_url
                    header, encoded = data_uri.split(",", 1)
                    image_bytes = base64.b64decode(encoded)
                    with open(image_path, "wb") as f:
                        f.write(image_bytes)
                else:
                    # Handle regular URL
                    response = requests.get(image_url, stream=True)
                    response.raise_for_status()

                    with open(image_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)

                output_images.append(File(path=image_path))

            self.logger.info(f"Successfully generated {len(output_images)} image(s)")

            # Prepare output with images and description
            return AppOutput(
                images=output_images,
                description=description
            )

        except Exception as e:
            self.logger.error(f"Error during image editing: {e}")
            raise RuntimeError(f"Image editing failed: {str(e)}")

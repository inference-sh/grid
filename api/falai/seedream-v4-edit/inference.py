from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field, BaseModel
from typing import Optional, List
import fal_client
import tempfile
import os
import logging
from PIL import Image
import requests


class AppInput(BaseAppInput):
    prompt: str = Field(
        description="The text prompt used to edit the image. Describe what you want to change or create in the image."
    )
    image_urls: List[File] = Field(
        description="List of input images for editing. Up to 10 images are allowed. If over 10 images are sent, only the last 10 will be used.",
        min_length=1,
        max_length=10
    )
    width: Optional[int] = Field(
        None,
        description="Width of the generated image in pixels. If not specified, uses the width of the first input image.",
        gt=0
    )
    height: Optional[int] = Field(
        None,
        description="Height of the generated image in pixels. If not specified, uses the height of the first input image.",
        gt=0
    )
    num_images: int = Field(
        1,
        description="Number of separate model generations to be run with the prompt.",
        ge=1,
        le=6
    )
    max_images: int = Field(
        1,
        description="If set to a number greater than one, enables multi-image generation. The model will potentially return up to max_images images every generation. The total number of images (inputs + outputs) must not exceed 15.",
        ge=1,
        le=6
    )
    seed: Optional[int] = Field(
        None,
        description="Random seed to control the stochasticity of image generation."
    )
    sync_mode: bool = Field(
        False,
        description="If True, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    enable_safety_checker: bool = Field(
        True,
        description="If set to true, the safety checker will be enabled."
    )


class AppOutput(BaseAppOutput):
    images: List[File] = Field(description="Generated images")
    seed: int = Field(description="Seed used for generation")


class App(BaseApp):
    async def setup(self, metadata):
        """Initialize model and configuration."""
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Store metadata for later use
        self.metadata = metadata

        # Model endpoint
        self.model_id = "fal-ai/bytedance/seedream/v4/edit"

        self.logger.info("Bytedance Seedream v4 Edit initialized successfully")

    def _get_image_dimensions(self, image_file: File) -> tuple[int, int]:
        """Get the width and height of an image file."""
        try:
            # Check if it's a URL
            if image_file.path.startswith("http://") or image_file.path.startswith("https://"):
                # Download the image temporarily to get dimensions
                response = requests.get(image_file.path, stream=True)
                response.raise_for_status()
                img = Image.open(response.raw)
            else:
                # Open local file
                img = Image.open(image_file.path)

            width, height = img.size
            self.logger.info(f"Detected image dimensions: {width}x{height}")
            return width, height
        except Exception as e:
            self.logger.error(f"Failed to get image dimensions: {e}")
            raise RuntimeError(f"Failed to get image dimensions: {e}")

    def _prepare_model_request(self, input_data: AppInput) -> dict:
        """Prepare the request payload for model inference."""
        # Upload all image files to get URLs

        # Prepare request data with required fields
        request_data = {
            "prompt": input_data.prompt,
            "image_urls": [image.uri for image in input_data.image_urls],
            "num_images": input_data.num_images,
            "max_images": input_data.max_images,
            "sync_mode": input_data.sync_mode,
            "enable_safety_checker": input_data.enable_safety_checker,
        }

        # Handle image size - use first input image dimensions if not specified
        if input_data.width is not None and input_data.height is not None:
            request_data["image_size"] = {
                "height": input_data.height,
                "width": input_data.width
            }
        else:
            # Get dimensions from the first input image
            first_image = input_data.image_urls[0]
            width, height = self._get_image_dimensions(first_image)
            request_data["image_size"] = {
                "height": height,
                "width": width
            }
            self.logger.info(f"Using first input image dimensions: {width}x{height}")

        if input_data.seed is not None:
            request_data["seed"] = input_data.seed

        return request_data

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Edit images using Bytedance Seedream v4 model."""
        try:
            # Validate input files (only if they're local paths)
            for i, image in enumerate(input_data.image_urls):
                if not image.path.startswith("http://") and not image.path.startswith("https://"):
                    if not image.exists():
                        raise RuntimeError(f"Input image {i+1} does not exist at path: {image.path}")

            # Set up API key from environment
            api_key = os.environ.get("FAL_KEY")
            if not api_key:
                raise RuntimeError(
                    "FAL_KEY environment variable is required for model access."
                )

            # Configure client with API key
            fal_client.api_key = api_key

            self.logger.info(f"Starting image editing with prompt: {input_data.prompt[:100]}...")
            self.logger.info(f"Processing {len(input_data.image_urls)} input image(s)")
            self.logger.info(f"Generating {input_data.num_images} output image(s)")
            if input_data.max_images > 1:
                self.logger.info(f"Multi-image generation enabled: up to {input_data.max_images} images per generation")

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
            self.logger.info(f"Generation seed: {result.get('seed', 'N/A')}")

            # Process the generated images
            output_images = []
            for i, image_data in enumerate(result["images"]):
                self.logger.info(f"Processing generated image {i+1}...")

                # Create temporary file for the image
                # Determine file extension from URL or use default .png
                image_url = image_data.get("url", "")
                if image_url.endswith(".jpg") or image_url.endswith(".jpeg"):
                    file_extension = ".jpg"
                else:
                    file_extension = ".png"

                with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as tmp_file:
                    image_path = tmp_file.name

                # Download image content
                import requests
                if input_data.sync_mode and "data:" in image_url:
                    # Handle data URI format
                    import base64
                    header, encoded = image_url.split(",", 1)
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

            self.logger.info(f"Image processing completed successfully. Generated {len(output_images)} image(s)")

            # Prepare output
            return AppOutput(
                images=output_images,
                seed=result["seed"]
            )

        except Exception as e:
            self.logger.error(f"Error during image editing: {e}")
            raise RuntimeError(f"Image editing failed: {str(e)}")


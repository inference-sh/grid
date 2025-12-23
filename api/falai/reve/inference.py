from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from typing import Optional, List
from enum import Enum
import fal_client
import tempfile
import os
import logging

class OutputFormatEnum(str, Enum):
    """Output format options."""
    png = "png"
    jpeg = "jpeg"
    webp = "webp"

class AppInput(BaseAppInput):
    prompt: str = Field(
        description="The text description of how to edit the provided image. Example: 'Give him a friend'"
    )
    image: File = Field(
        description="The reference image to edit. Supported formats: PNG, JPEG, WebP, AVIF, and HEIF."
    )
    output_format: OutputFormatEnum = Field(
        OutputFormatEnum.png,
        description="Output format for the generated image. Default: png"
    )
    sync_mode: bool = Field(
        False,
        description="If True, the media will be returned as a data URI and the output data won't be available in the request history."
    )

class AppOutput(BaseAppOutput):
    images: List[File] = Field(description="The edited image")

class App(BaseApp):
    async def setup(self, metadata):
        """Initialize model and configuration."""
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Store metadata for later use
        self.metadata = metadata

        # Model endpoint
        self.model_id = "fal-ai/reve/edit"

        self.logger.info("Reve Image Editor initialized successfully")

    def _upload_file_to_url(self, file_path: str) -> str:
        """Upload a local file to temporary storage for processing."""
        try:
            # Upload file and get a temporary URL
            file_url = fal_client.upload_file(file_path)
            self.logger.info(f"File uploaded to temporary storage successfully")
            return file_url
        except Exception as e:
            self.logger.error(f"Failed to upload file {file_path}: {e}")
            raise RuntimeError(f"Failed to upload file: {e}")

    def _prepare_model_request(self, input_data: AppInput) -> dict:
        """Prepare the request payload for model inference."""
        # Upload image file to get URL
        image_url = self._upload_file_to_url(input_data.image.path)

        # Prepare request data
        request_data = {
            "prompt": input_data.prompt,
            "image_url": image_url,
            "output_format": input_data.output_format.value,
            "sync_mode": input_data.sync_mode,
        }

        return request_data

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Edit image using Reve model."""
        try:
            # Validate input file
            if not input_data.image.exists():
                raise RuntimeError(f"Input image does not exist at path: {input_data.image.path}")

            # Set up API key from environment
            api_key = os.environ.get("FAL_KEY")
            if not api_key:
                raise RuntimeError(
                    "FAL_KEY environment variable is required for model access."
                )

            # Configure client with API key
            fal_client.api_key = api_key

            self.logger.info(f"Starting image editing with prompt: {input_data.prompt[:100]}...")
            self.logger.info(f"Processing input image: {input_data.image.path}")

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

            # Process the generated images
            output_images = []
            for i, image_data in enumerate(result["images"]):
                self.logger.info(f"Processing generated image {i+1}...")

                # Create temporary file for the image
                file_extension = f".{input_data.output_format.value}"
                with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as tmp_file:
                    image_path = tmp_file.name

                # Download image content
                import requests
                if input_data.sync_mode and "data:" in image_data.get("url", ""):
                    # Handle data URI format
                    import base64
                    data_uri = image_data["url"]
                    header, encoded = data_uri.split(",", 1)
                    image_bytes = base64.b64decode(encoded)
                    with open(image_path, "wb") as f:
                        f.write(image_bytes)
                else:
                    # Handle regular URL
                    image_url = image_data["url"]
                    response = requests.get(image_url, stream=True)
                    response.raise_for_status()

                    with open(image_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)

                output_images.append(File(path=image_path))

            self.logger.info(f"Image processing completed successfully")

            # Prepare output
            return AppOutput(
                images=output_images
            )

        except Exception as e:
            self.logger.error(f"Error during image editing: {e}")
            raise RuntimeError(f"Image editing failed: {str(e)}")


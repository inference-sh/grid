from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from typing import Optional
from enum import Enum
import fal_client
import tempfile
import os
import logging

class ModelEnum(str, Enum):
    """Model options for Topaz image enhancement."""
    low_resolution_v2 = "Low Resolution V2"
    standard_v2 = "Standard V2"
    cgi = "CGI"
    high_fidelity_v2 = "High Fidelity V2"
    text_refine = "Text Refine"
    recovery = "Recovery"
    redefine = "Redefine"
    recovery_v2 = "Recovery V2"

class OutputFormatEnum(str, Enum):
    """Output format options."""
    jpeg = "jpeg"
    png = "png"

class SubjectDetectionEnum(str, Enum):
    """Subject detection mode options."""
    all = "All"
    foreground = "Foreground"
    background = "Background"

class AppInput(BaseAppInput):
    image: File = Field(
        description="Input image file to be upscaled and enhanced"
    )
    model: ModelEnum = Field(
        ModelEnum.standard_v2,
        description="Model to use for image enhancement"
    )
    upscale_factor: float = Field(
        2.0,
        ge=1.0,
        le=4.0,
        description="Factor to upscale the image by (e.g. 2.0 doubles width and height)"
    )
    crop_to_fill: bool = Field(
        False,
        description="Whether to crop the image to fill the output dimensions"
    )
    output_format: OutputFormatEnum = Field(
        OutputFormatEnum.jpeg,
        description="Output format of the upscaled image"
    )
    subject_detection: SubjectDetectionEnum = Field(
        SubjectDetectionEnum.all,
        description="Subject detection mode for the image enhancement"
    )
    face_enhancement: bool = Field(
        True,
        description="Whether to apply face enhancement to the image"
    )
    face_enhancement_creativity: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Creativity level for face enhancement. 0.0 means no creativity, 1.0 means maximum creativity. Ignored if face enhancement is disabled."
    )
    face_enhancement_strength: float = Field(
        0.8,
        ge=0.0,
        le=1.0,
        description="Strength of the face enhancement. 0.0 means no enhancement, 1.0 means maximum enhancement. Ignored if face enhancement is disabled."
    )

class AppOutput(BaseAppOutput):
    image: File = Field(description="The upscaled and enhanced image")

class App(BaseApp):
    async def setup(self, metadata):
        """Initialize model and configuration."""
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Store metadata for later use
        self.metadata = metadata

        # Model endpoint
        self.model_id = "fal-ai/topaz/upscale/image"

        self.logger.info("Topaz Image Upscaler initialized successfully")


    def _prepare_model_request(self, input_data: AppInput) -> dict:
        """Prepare the request payload for model inference."""
        # Upload image file to get URL

        # Prepare request data
        request_data = {
            "image_url": input_data.image.uri,
            "model": input_data.model.value,
            "upscale_factor": input_data.upscale_factor,
            "crop_to_fill": input_data.crop_to_fill,
            "output_format": input_data.output_format.value,
            "subject_detection": input_data.subject_detection.value,
            "face_enhancement": input_data.face_enhancement,
        }

        # Add face enhancement parameters only if face enhancement is enabled
        if input_data.face_enhancement:
            request_data["face_enhancement_creativity"] = input_data.face_enhancement_creativity
            request_data["face_enhancement_strength"] = input_data.face_enhancement_strength

        return request_data

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Upscale and enhance image using Topaz model."""
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

            self.logger.info(f"Starting image upscaling with model: {input_data.model.value}")
            self.logger.info(f"Upscale factor: {input_data.upscale_factor}x")
            self.logger.info(f"Face enhancement: {'enabled' if input_data.face_enhancement else 'disabled'}")

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

            self.logger.info("Image upscaling completed successfully")

            # Process the generated image
            image_url = result["image"]["url"]
            self.logger.info("Processing enhanced image output...")

            # Determine output file extension
            ext = f".{input_data.output_format.value}"

            # Create temporary file for the image
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_file:
                image_path = tmp_file.name

            # Download image content
            import requests
            response = requests.get(image_url, stream=True)
            response.raise_for_status()

            with open(image_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            self.logger.info(f"Image processing completed successfully")

            # Prepare output
            return AppOutput(
                image=File(path=image_path)
            )

        except Exception as e:
            self.logger.error(f"Error during image upscaling: {e}")
            raise RuntimeError(f"Image upscaling failed: {str(e)}")


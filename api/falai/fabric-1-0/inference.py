from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from typing import Optional
from enum import Enum
import fal_client
import tempfile
import os
import logging

class ResolutionEnum(str, Enum):
    """Resolution options for video generation."""
    p480 = "480p"
    p720 = "720p"

class AppInput(BaseAppInput):
    image: File = Field(
        description="Image to turn into a talking video. Supported formats: JPEG, PNG, WebP"
    )
    audio: File = Field(
        description="Audio file for the talking video. Supported formats: WAV, MP3"
    )
    resolution: ResolutionEnum = Field(
        ResolutionEnum.p480,
        description="Video resolution. Higher resolutions provide better quality but take longer to generate."
    )

class AppOutput(BaseAppOutput):
    video: File = Field(description="Generated talking video with lip-sync animation")

class App(BaseApp):
    async def setup(self, metadata):
        """Initialize model and configuration."""
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Store metadata for later use
        self.metadata = metadata

        # Model endpoint
        self.model_id = "veed/fabric-1.0"

        self.logger.info("Fabric 1.0 model initialized successfully")

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Generate talking video using Fabric 1.0 model."""
        try:
          
            # Set up API key from environment
            api_key = os.environ.get("FAL_KEY")
            if not api_key:
                raise RuntimeError(
                    "FAL_KEY environment variable is required for model access."
                )

            # Configure client with API key
            fal_client.api_key = api_key

            self.logger.info("Starting image-to-talking-video generation...")
            self.logger.info(f"Resolution: {input_data.resolution.value}")

            # Prepare request data
            request_data = {
                "image_url": input_data.image.uri,
                "audio_url": input_data.audio.uri,
                "resolution": input_data.resolution.value,
            }

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

            self.logger.info("Talking video generation completed successfully")

            # Process the generated video
            video_url = result["video"]["url"]
            self.logger.info("Processing generated video output...")

            # Create temporary file for the video
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
                video_path = tmp_file.name

            # Download video content
            import requests
            response = requests.get(video_url, stream=True)
            response.raise_for_status()

            with open(video_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            self.logger.info(f"Video processing completed successfully")

            # Prepare output
            return AppOutput(
                video=File(path=video_path)
            )

        except Exception as e:
            self.logger.error(f"Error during talking video generation: {e}")
            raise RuntimeError(f"Talking video generation failed: {str(e)}")


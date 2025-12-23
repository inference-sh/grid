from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from typing import Optional
import fal_client
import tempfile
import os
import logging

class AppInput(BaseAppInput):
    video: File = Field(
        description="Input video file to be upscaled"
    )
    upscale_factor: float = Field(
        2.0,
        ge=1.0,
        le=4.0,
        description="Factor to upscale the video by (e.g. 2.0 doubles width and height)"
    )
    target_fps: Optional[int] = Field(
        None,
        ge=16,
        le=60,
        description="Target frames per second for interpolation"
    )
    H264_output: bool = Field(
        False,
        description="Use H264 codec instead of H265 (default is H265)"
    )

class AppOutput(BaseAppOutput):
    video: File = Field(description="The upscaled video")

class App(BaseApp):
    async def setup(self, metadata):
        """Initialize model and configuration."""
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Store metadata for later use
        self.metadata = metadata

        # Model endpoint
        self.model_id = "fal-ai/topaz/upscale/video"

        self.logger.info("Topaz Video Upscaler initialized successfully")

    def _upload_file_to_url(self, file_path: str) -> str:
        """Upload a local file to temporary storage for processing."""
        try:
            # Upload file and get a temporary URL
            file_url = fal_client.upload_file(file_path)
            self.logger.info(f"Video uploaded to temporary storage successfully")
            return file_url
        except Exception as e:
            self.logger.error(f"Failed to upload file {file_path}: {e}")
            raise RuntimeError(f"Failed to upload file: {e}")

    def _prepare_model_request(self, input_data: AppInput) -> dict:
        """Prepare the request payload for model inference."""
        # Upload video file to get URL
        video_url = self._upload_file_to_url(input_data.video.path)

        # Prepare request data
        request_data = {
            "video_url": video_url,
            "upscale_factor": input_data.upscale_factor,
            "H264_output": input_data.H264_output,
        }

        # Add target_fps if specified
        if input_data.target_fps is not None:
            request_data["target_fps"] = input_data.target_fps

        return request_data

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Upscale video using Topaz model."""
        try:
            # Validate input file
            if not input_data.video.exists():
                raise RuntimeError(f"Input video does not exist at path: {input_data.video.path}")

            # Set up API key from environment
            api_key = os.environ.get("FAL_KEY")
            if not api_key:
                raise RuntimeError(
                    "FAL_KEY environment variable is required for model access."
                )

            # Configure client with API key
            fal_client.api_key = api_key

            self.logger.info(f"Starting video upscaling...")
            self.logger.info(f"Upscale factor: {input_data.upscale_factor}x")
            if input_data.target_fps:
                self.logger.info(f"Target FPS: {input_data.target_fps}")
            self.logger.info(f"Codec: {'H264' if input_data.H264_output else 'H265'}")

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

            self.logger.info("Video upscaling completed successfully")

            # Process the generated video
            video_url = result["video"]["url"]
            self.logger.info("Processing upscaled video output...")

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
            self.logger.error(f"Error during video upscaling: {e}")
            raise RuntimeError(f"Video upscaling failed: {str(e)}")


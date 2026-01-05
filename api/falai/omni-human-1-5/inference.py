from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta
from pydantic import Field
from typing import Optional
import fal_client
import tempfile
import os
import logging

class AppInput(BaseAppInput):
    image: File = Field(
        description="Image of a human figure used to generate the video. Supported formats: JPEG, PNG, WebP"
    )
    audio: File = Field(
        description="Audio file to generate the video. Must be under 30 seconds long. Supported formats: WAV, MP3"
    )

class AppOutput(BaseAppOutput):
    video: File = Field(description="Generated video file with synchronized lip movements and emotions")
    duration: float = Field(description="Duration of audio input/video output in seconds")

class App(BaseApp):
    async def setup(self, metadata):
        """Initialize model and configuration."""
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Store metadata for later use
        self.metadata = metadata

        # Model endpoint
        self.model_id = "fal-ai/bytedance/omnihuman/v1.5"

        self.logger.info("Omni-Human 1.5 model initialized successfully")

    def _prepare_model_request(self, input_data: AppInput) -> dict:
        """Prepare the request payload for model inference."""
        # Upload image and audio files to get URLs

        # Prepare request data
        request_data = {
            "image_url": input_data.image.uri,
            "audio_url": input_data.audio.uri,
        }

        return request_data

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Generate audio-driven video using Omni-Human 1.5 model."""
        try:
            # Validate input files
            if not input_data.image.exists():
                raise RuntimeError(f"Input image does not exist at path: {input_data.image.path}")

            if not input_data.audio.exists():
                raise RuntimeError(f"Input audio does not exist at path: {input_data.audio.path}")

            # Set up API key from environment
            api_key = os.environ.get("FAL_KEY")
            if not api_key:
                raise RuntimeError(
                    "FAL_KEY environment variable is required for model access."
                )

            # Configure client with API key
            fal_client.api_key = api_key

            self.logger.info("Starting audio-driven video generation...")
            self.logger.info("Processing human image and audio for lip-sync video")

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

            self.logger.info("Video generation completed successfully")

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

            # Build output metadata for pricing
            output_meta = OutputMeta(
                outputs=[
                    VideoMeta(
                        seconds=result["duration"]
                    )
                ]
            )

            # Prepare output
            return AppOutput(
                video=File(path=video_path),
                duration=result["duration"],
                output_meta=output_meta
            )

        except Exception as e:
            self.logger.error(f"Error during video generation: {e}")
            raise RuntimeError(f"Video generation failed: {str(e)}")


from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from typing import Optional
from enum import Enum
import fal_client
import tempfile
import os
import logging

class VoiceIdEnum(str, Enum):
    """Voice options for text-to-speech."""
    emily = "Emily"
    james = "James"
    isabella = "Isabella"
    liam = "Liam"
    chloe = "Chloe"
    adrian = "Adrian"
    harper = "Harper"
    ava = "Ava"
    sophia = "Sophia"
    julia = "Julia"
    mason = "Mason"
    jack = "Jack"
    oliver = "Oliver"
    ethan = "Ethan"
    auto = "Auto"

class AppInput(BaseAppInput):
    video: File = Field(
        description="Input video file for lip synchronization. Supported formats: MP4, MOV, AVI"
    )
    audio: Optional[File] = Field(
        None,
        description="Audio file for lip sync. If not provided, TTS will be used with the text parameter. Supported formats: WAV, MP3"
    )
    voice_id: VoiceIdEnum = Field(
        VoiceIdEnum.auto,
        description="Voice to use for TTS when audio is not provided."
    )
    text: Optional[str] = Field(
        None,
        description="Text content for TTS when audio is not provided. Required if no audio file is given."
    )

class AppOutput(BaseAppOutput):
    video: File = Field(description="Generated video with synchronized lip movements")

class App(BaseApp):
    async def setup(self, metadata):
        """Initialize model and configuration."""
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Store metadata for later use
        self.metadata = metadata

        # Model endpoint
        self.model_id = "fal-ai/pixverse/lipsync"

        self.logger.info("Pixverse Lipsync model initialized successfully")

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
        # Upload video file to get URL
        video_url = self._upload_file_to_url(input_data.video.path)

        # Prepare base request
        request_data = {
            "video_url": video_url,
        }

        # Add audio if provided
        if input_data.audio:
            audio_url = self._upload_file_to_url(input_data.audio.path)
            request_data["audio_url"] = audio_url
        else:
            # Use TTS parameters
            request_data["voice_id"] = input_data.voice_id.value
            if input_data.text:
                request_data["text"] = input_data.text

        return request_data

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Generate lip-synced video using Pixverse Lipsync model."""
        try:
            # Validate input files
            if not input_data.video.exists():
                raise RuntimeError(f"Input video does not exist at path: {input_data.video.path}")

            if input_data.audio and not input_data.audio.exists():
                raise RuntimeError(f"Input audio does not exist at path: {input_data.audio.path}")

            # Validate TTS parameters if no audio provided
            if not input_data.audio and not input_data.text:
                raise RuntimeError("Either audio file or text for TTS must be provided")

            # Set up API key from environment
            api_key = os.environ.get("FAL_KEY")
            if not api_key:
                raise RuntimeError(
                    "FAL_KEY environment variable is required for model access."
                )

            # Configure client with API key
            fal_client.api_key = api_key

            if input_data.audio:
                self.logger.info("Starting video lip-sync with provided audio...")
            else:
                self.logger.info(f"Starting video lip-sync with TTS using voice: {input_data.voice_id.value}")
                self.logger.info(f"TTS text: {input_data.text[:100]}...")

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

            self.logger.info("Lip-sync generation completed successfully")

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
            self.logger.error(f"Error during lip-sync generation: {e}")
            raise RuntimeError(f"Lip-sync generation failed: {str(e)}")

    async def unload(self):
        """Clean up resources."""
        self.logger.info("Pixverse Lipsync model unloaded successfully")
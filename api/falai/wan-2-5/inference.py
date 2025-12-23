from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from typing import Optional
from enum import Enum
import fal_client
import tempfile
import os
import logging

class ResolutionEnum(str, Enum):
    """Video resolution options."""
    p480 = "480p"
    p720 = "720p"
    p1080 = "1080p"

class AppInput(BaseAppInput):
    prompt: str = Field(
        description="The text prompt describing the desired video motion. Max 800 characters.",
        max_length=800
    )
    image: File = Field(
        description="Image file to use as the first frame. Supported formats: JPEG, PNG, WebP"
    )
    audio: Optional[File] = Field(
        None,
        description="Optional audio file to use as background music. Supported formats: WAV, MP3. Duration: 3-30s, max 15MB. Audio will be truncated if longer than video duration."
    )
    resolution: ResolutionEnum = Field(
        ResolutionEnum.p1080,
        description="Video resolution. Higher resolutions provide better quality but take longer to generate."
    )
    negative_prompt: Optional[str] = Field(
        None,
        description="Negative prompt to describe content to avoid. Max 500 characters.",
        max_length=500
    )
    enable_prompt_expansion: bool = Field(
        True,
        description="Whether to enable prompt rewriting using LLM for better results."
    )
    seed: Optional[int] = Field(
        None,
        description="Random seed for reproducibility. If None, a random seed is chosen."
    )
    fal_api_key: Optional[str] = Field(
        None,
        description="fal.ai API key. If not provided, will look for FAL_KEY environment variable."
    )

class AppOutput(BaseAppOutput):
    video: File = Field(description="The generated video file")
    seed: int = Field(description="The seed used for generation")
    actual_prompt: Optional[str] = Field(
        None,
        description="The actual prompt used if prompt rewriting was enabled"
    )

# For LLM apps, you can use the LLMInput and LLMInputWithImage classes for convenience
# from inferencesh import LLMInput, LLMInputWithImage
# The LLMInput class provides a standard structure for LLM-based applications with:
# - system_prompt: Sets the AI assistant's role and behavior
# - context: List of previous conversation messages between user and assistant
# - text: The current user's input prompt
#
# Example usage:
# class AppInput(LLMInput):
#     additional_field: str = Field(description="Any additional input needed")

# The LLMInputWithImage class extends LLMInput to support image inputs by adding:
# - image: Optional File field for providing images to vision-capable models
#
# Example usage:
# class AppInput(LLMInputWithImage):
#     additional_field: str = Field(description="Any additional input needed")

# Each ContextMessage in the context list contains:
# - role: Either "user", "assistant", or "system"
# - text: The message content
#
# ContextMessageWithImage adds:
# - image: Optional File field for messages containing images



class App(BaseApp):
    async def setup(self, metadata):
        """Initialize fal.ai client and configuration."""
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Store metadata for later use
        self.metadata = metadata

        # fal.ai model endpoint
        self.model_id = "fal-ai/wan-25-preview/image-to-video"

        self.logger.info(f"Wan 2.5 Image-to-Video app initialized with model: {self.model_id}")

    def _prepare_fal_request(self, input_data: AppInput) -> dict:
        """Prepare the request payload for fal.ai API."""
        # Prepare base request
        request_data = {
            "prompt": input_data.prompt,
            "image_url": input_data.image.uri,
            "resolution": input_data.resolution.value,
            "enable_prompt_expansion": input_data.enable_prompt_expansion,
        }

        # Add optional parameters
        if input_data.audio:
            request_data["audio_url"] = input_data.audio.uri

        if input_data.negative_prompt:
            request_data["negative_prompt"] = input_data.negative_prompt

        if input_data.seed is not None:
            request_data["seed"] = input_data.seed

        return request_data

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Generate video using fal.ai Wan 2.5 model."""
        try:
            # Validate input files
            if not input_data.image.exists():
                raise RuntimeError(f"Input image does not exist at path: {input_data.image.path}")

            if input_data.audio and not input_data.audio.exists():
                raise RuntimeError(f"Input audio does not exist at path: {input_data.audio.path}")

            # Set up fal.ai API key
            api_key = input_data.fal_api_key or os.environ.get("FAL_KEY")
            if not api_key:
                raise RuntimeError(
                    "fal.ai API key is required. Either provide it in the request or set the FAL_KEY environment variable."
                )

            # Configure fal_client with API key
            fal_client.api_key = api_key

            self.logger.info(f"Starting video generation with prompt: {input_data.prompt[:100]}...")
            self.logger.info(f"Resolution: {input_data.resolution.value}")

            # Prepare request data for fal.ai
            request_data = self._prepare_fal_request(input_data)

            self.logger.info("Sending request to fal.ai Wan 2.5 model...")

            # Define progress callback
            def on_queue_update(update):
                if isinstance(update, fal_client.InProgress):
                    for log in update.logs:
                        self.logger.info(f"fal.ai: {log['message']}")

            # Call fal.ai API with progress logging
            result = fal_client.subscribe(
                self.model_id,
                arguments=request_data,
                with_logs=True,
                on_queue_update=on_queue_update,
            )

            self.logger.info("Video generation completed successfully")

            # Download the generated video
            video_url = result["video"]["url"]
            self.logger.info(f"Downloading generated video from: {video_url}")

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

            self.logger.info(f"Video downloaded successfully to: {video_path}")

            # Prepare output
            return AppOutput(
                video=File(path=video_path),
                seed=result["seed"],
                actual_prompt=result.get("actual_prompt")
            )

        except Exception as e:
            self.logger.error(f"Error during video generation: {e}")
            raise RuntimeError(f"Video generation failed: {str(e)}")


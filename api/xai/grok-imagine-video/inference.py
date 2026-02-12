"""
Grok Imagine Video - xAI Video Generation

Generate and edit videos using xAI's Grok Imagine Video model.
Supports text-to-video, image-to-video, and video editing.
"""

from typing import Optional, Literal
import tempfile

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta, VideoResolution
from pydantic import Field
import requests

from .xai_helper import (
    VideoAspectRatioType,
    create_xai_client,
    setup_logger,
    encode_image_base64,
    get_video_dimensions,
    retry_on_rate_limit,
)


ResolutionType = Literal["720p", "480p"]


class AppInput(BaseAppInput):
    """Input schema for Grok Imagine video generation."""

    prompt: str = Field(
        description="Text prompt describing the video content and motion.",
        examples=["A cat playing with a ball", "A serene forest with gentle wind moving the leaves"]
    )
    image: Optional[File] = Field(
        default=None,
        description="Optional input image for image-to-video generation. The video will animate from this starting frame."
    )
    video: Optional[File] = Field(
        default=None,
        description="Optional input video for video editing. The model will edit this video based on the prompt. Max 8.7 seconds."
    )
    duration: int = Field(
        default=5,
        ge=1,
        le=15,
        description="Duration of the generated video in seconds (1-15). Not applicable for video editing."
    )
    aspect_ratio: VideoAspectRatioType = Field(
        default="16:9",
        description="Aspect ratio of the generated video."
    )
    resolution: ResolutionType = Field(
        default="720p",
        description="Video resolution. 720p for higher quality, 480p for faster generation."
    )


class AppOutput(BaseAppOutput):
    """Output schema for Grok Imagine video generation."""

    video: File = Field(description="The generated video file.")


class App(BaseApp):
    """Grok Imagine video generation application using xAI SDK."""

    async def setup(self, metadata):
        """Initialize the xAI client."""
        self.logger = setup_logger(__name__)
        self.client = create_xai_client()
        self.model = "grok-imagine-video"
        self.logger.info(f"Grok Imagine Video initialized with model: {self.model}")

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Generate or edit video using Grok Imagine Video."""
        try:
            # Determine mode
            if input_data.video:
                mode = "video-edit"
            elif input_data.image:
                mode = "image-to-video"
            else:
                mode = "text-to-video"

            self.logger.info(f"Starting {mode} generation")
            self.logger.info(f"Prompt: {input_data.prompt[:100]}...")
            self.logger.info(f"Duration: {input_data.duration}s, Resolution: {input_data.resolution}, Aspect ratio: {input_data.aspect_ratio}")

            # Build kwargs for the API call
            kwargs = {
                "model": self.model,
                "prompt": input_data.prompt,
                "aspect_ratio": input_data.aspect_ratio,
                "resolution": input_data.resolution,
            }

            # Add duration for non-edit modes
            if mode != "video-edit":
                kwargs["duration"] = input_data.duration

            # Handle image input (image-to-video)
            if input_data.image:
                if input_data.image.uri and input_data.image.uri.startswith("http"):
                    kwargs["image_url"] = input_data.image.uri
                else:
                    kwargs["image_url"] = encode_image_base64(input_data.image)

            # Handle video input (video editing)
            if input_data.video:
                if not input_data.video.exists():
                    raise RuntimeError(f"Input video does not exist at path: {input_data.video.path}")

                if input_data.video.uri and input_data.video.uri.startswith("http"):
                    kwargs["video_url"] = input_data.video.uri
                else:
                    raise RuntimeError("Video editing requires a publicly accessible video URL.")

            # Generate video (SDK handles polling automatically, with 429 retry)
            self.logger.info("Starting video generation (SDK will poll automatically)...")
            response = await retry_on_rate_limit(
                lambda: self.client.video.generate(**kwargs),
                logger=self.logger,
            )

            # Download video from URL
            video_url = response.url
            if not video_url:
                raise RuntimeError("No video URL in response")

            self.logger.info(f"Downloading video from: {video_url}")
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                video_response = requests.get(video_url, timeout=300)
                video_response.raise_for_status()
                f.write(video_response.content)
                video_path = f.name

            # Get duration from response or use input
            duration_seconds = getattr(response, 'duration', float(input_data.duration))

            # Map resolution to enum and get dimensions
            resolution_enum_map = {
                "480p": VideoResolution.RES_480P,
                "720p": VideoResolution.RES_720P,
            }
            resolution_enum = resolution_enum_map.get(input_data.resolution, VideoResolution.RES_720P)
            width, height = get_video_dimensions(input_data.aspect_ratio, input_data.resolution)

            output_meta = OutputMeta(
                outputs=[
                    VideoMeta(
                        width=width,
                        height=height,
                        resolution=resolution_enum,
                        seconds=float(duration_seconds),
                        fps=24,
                        extra={
                            "mode": mode,
                            "aspect_ratio": input_data.aspect_ratio,
                        }
                    )
                ]
            )

            self.logger.info(f"Video generated successfully: {video_path}")

            return AppOutput(
                video=File(path=video_path),
                output_meta=output_meta,
            )

        except Exception as e:
            self.logger.error(f"Error during video generation: {e}")
            raise RuntimeError(f"Video generation failed: {str(e)}")

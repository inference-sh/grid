"""
Grok Imagine Video - xAI Video Generation

Generate and edit videos using xAI's Grok Imagine Video model.
Supports text-to-video, image-to-video, and video editing.
"""

import base64
import os
import logging
import tempfile
from typing import Optional, Literal

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta, VideoResolution
from pydantic import Field
from xai_sdk import Client
import requests


AspectRatioType = Literal["16:9", "4:3", "1:1", "9:16", "3:4", "3:2", "2:3"]
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
    aspect_ratio: AspectRatioType = Field(
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
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        api_key = os.environ.get("XAI_API_KEY")
        if not api_key:
            raise RuntimeError("XAI_API_KEY environment variable is required")

        self.client = Client(api_key=api_key)
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
                if not input_data.image.exists():
                    raise RuntimeError(f"Input image does not exist at path: {input_data.image.path}")

                # Use URL if available, otherwise encode as base64
                if input_data.image.uri and input_data.image.uri.startswith("http"):
                    kwargs["image_url"] = input_data.image.uri
                else:
                    with open(input_data.image.path, "rb") as f:
                        image_bytes = f.read()
                        base64_string = base64.b64encode(image_bytes).decode("utf-8")
                    content_type = input_data.image.content_type or "image/jpeg"
                    kwargs["image_url"] = f"data:{content_type};base64,{base64_string}"

            # Handle video input (video editing)
            if input_data.video:
                if not input_data.video.exists():
                    raise RuntimeError(f"Input video does not exist at path: {input_data.video.path}")

                if input_data.video.uri and input_data.video.uri.startswith("http"):
                    kwargs["video_url"] = input_data.video.uri
                else:
                    raise RuntimeError("Video editing requires a publicly accessible video URL.")

            # Generate video (SDK handles polling automatically)
            self.logger.info("Starting video generation (SDK will poll automatically)...")
            response = self.client.video.generate(**kwargs)

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

            # Map resolution to enum and dimensions
            resolution_map = {
                "480p": (VideoResolution.RES_480P, 854, 480),
                "720p": (VideoResolution.RES_720P, 1280, 720),
            }
            resolution_enum, base_width, base_height = resolution_map.get(
                input_data.resolution, (VideoResolution.RES_720P, 1280, 720)
            )

            # Adjust dimensions based on aspect ratio
            aspect_map = {
                "16:9": (16, 9),
                "4:3": (4, 3),
                "1:1": (1, 1),
                "9:16": (9, 16),
                "3:4": (3, 4),
                "3:2": (3, 2),
                "2:3": (2, 3),
            }
            aspect_w, aspect_h = aspect_map.get(input_data.aspect_ratio, (16, 9))

            if aspect_w > aspect_h:
                width = base_width
                height = int(base_width * aspect_h / aspect_w)
            else:
                height = base_height
                width = int(height * aspect_w / aspect_h)

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

"""
Grok Extend Video - xAI Video Extension

Extend existing videos using xAI's Grok Imagine Video model.
Takes an existing video and generates additional frames to continue it.
"""

from typing import Optional, Literal

import tempfile

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta, VideoResolution, ImageMeta
from pydantic import Field
import requests

from .xai_helper import (
    XAIError,
    create_xai_client,
    setup_logger,
    retry_on_rate_limit,
    get_video_dimensions,
)


class AppInput(BaseAppInput):
    """Input schema for Grok video extension."""

    prompt: str = Field(
        description="Text prompt describing what should happen in the extended portion of the video.",
        examples=["The camera continues to pan across the landscape", "The person turns and walks away"]
    )
    video: File = Field(
        description="Input video to extend. Must be a publicly accessible URL."
    )
    duration: Optional[int] = Field(
        default=None,
        ge=1,
        le=15,
        description="Duration of the extended video in seconds (1-15). If not specified, defaults to model's default."
    )


class AppOutput(BaseAppOutput):
    """Output schema for Grok video extension."""

    video: File = Field(description="The extended video file.")


class App(BaseApp):
    """Grok video extension application using xAI SDK."""

    async def setup(self):
        """Initialize the xAI client."""
        self.logger = setup_logger(__name__)
        self.client = create_xai_client()
        self.model = "grok-imagine-video"
        self.logger.info(f"Grok Extend Video initialized with model: {self.model}")

    async def run(self, input_data: AppInput) -> AppOutput:
        """Extend a video using Grok Imagine Video."""
        try:
            self.logger.info(f"Starting video extension")
            self.logger.info(f"Prompt: {input_data.prompt[:100]}...")

            # Video must be a URL for extension
            if not input_data.video.uri or not input_data.video.uri.startswith("http"):
                raise RuntimeError("Video extension requires a publicly accessible video URL.")

            video_url = input_data.video.uri

            kwargs = {
                "prompt": input_data.prompt,
                "model": self.model,
                "video_url": video_url,
            }

            if input_data.duration is not None:
                kwargs["duration"] = input_data.duration
                self.logger.info(f"Duration: {input_data.duration}s")

            # Extend video (SDK handles polling automatically, with 429 retry)
            self.logger.info("Starting video extension (SDK will poll automatically)...")
            response = await retry_on_rate_limit(
                lambda: self.client.video.extend(**kwargs),
                logger=self.logger,
            )

            # Download video from URL
            video_url = response.url
            if not video_url:
                raise RuntimeError("No video URL in response")

            self.logger.info(f"Downloading extended video from: {video_url}")
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                video_response = requests.get(video_url, timeout=300)
                video_response.raise_for_status()
                f.write(video_response.content)
                video_path = f.name

            # Get duration from response
            duration_seconds = getattr(response, 'duration', float(input_data.duration or 8))

            # Track input video for billing (video input: $0.01/s)
            input_video_seconds = 0.0
            if input_data.video and input_data.video.exists():
                try:
                    import subprocess
                    result = subprocess.run(
                        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                         "-of", "default=noprint_wrappers=1:nokey=1", input_data.video.path],
                        capture_output=True, text=True, timeout=10
                    )
                    input_video_seconds = float(result.stdout.strip())
                except Exception as e:
                    self.logger.warning(f"Could not determine input video duration: {e}")

            output_meta = OutputMeta(
                inputs=[
                    VideoMeta(
                        width=0,
                        height=0,
                        resolution=VideoResolution.VIDEO_RES480_P,
                        seconds=input_video_seconds,
                        fps=24,
                        extra={"type": "video_input"}
                    )
                ],
                outputs=[
                    VideoMeta(
                        width=0,
                        height=0,
                        resolution=VideoResolution.VIDEO_RES480_P,
                        seconds=float(duration_seconds),
                        fps=24,
                        extra={
                            "mode": "extend",
                        }
                    )
                ]
            )

            self.logger.info(f"Video extended successfully: {video_path}")

            return AppOutput(
                video=File(path=video_path),
                output_meta=output_meta,
            )

        except XAIError:
            raise
        except Exception as e:
            self.logger.error(f"Error during video extension: {e}")
            raise RuntimeError(f"Video extension failed: {str(e)}")

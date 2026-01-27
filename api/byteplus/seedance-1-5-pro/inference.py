"""
Seedance 1.5 Pro - BytePlus Video Generation

Generate high-quality videos from text prompts with optional first-frame image control.
Uses BytePlus ARK SDK with async task polling.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta, VideoResolution
from pydantic import Field
from typing import Optional
from enum import Enum
import logging
import os

from .byteplus_helper import (
    setup_byteplus_client,
    create_content_task,
    poll_task_status,
    cancel_task,
    download_video,
    build_text_content,
    build_image_content,
)


class DurationEnum(str, Enum):
    """Video duration options in seconds."""
    s4 = "4"
    s5 = "5"
    s6 = "6"
    s7 = "7"
    s8 = "8"
    s9 = "9"
    s10 = "10"


class AppInput(BaseAppInput):
    """Input schema for Seedance 1.5 Pro video generation."""

    prompt: str = Field(
        description="Text prompt describing the video content and motion. Be descriptive about actions and camera movements.",
        examples=["At breakneck speed, drones thread through intricate obstacles, delivering an immersive flying experience."]
    )
    image: Optional[File] = Field(
        default=None,
        description="Optional first-frame image for image-to-video generation. If not provided, generates from text only."
    )
    duration: DurationEnum = Field(
        default=DurationEnum.s5,
        description="Duration of the video in seconds (4-10 seconds)."
    )
    camera_fixed: bool = Field(
        default=False,
        description="Whether to fix the camera position during video generation. Set to true for static camera shots."
    )


class AppOutput(BaseAppOutput):
    """Output schema for Seedance 1.5 Pro video generation."""

    video: File = Field(description="The generated video file.")


class App(BaseApp):
    """Seedance 1.5 Pro video generation application using BytePlus ARK SDK."""

    async def setup(self, metadata):
        """Initialize the BytePlus client."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Suppress noisy httpx logs
        logging.getLogger("httpx").setLevel(logging.WARNING)

        # Initialize client
        self.client = setup_byteplus_client()
        self.model_id = "seedance-1-5-pro-251215"

        # Cancellation support
        self.cancel_flag = False
        self.current_task_id = None

        self.logger.info(f"Seedance 1.5 Pro initialized with model: {self.model_id}")

    async def on_cancel(self):
        """Handle cancellation request."""
        self.logger.info("Cancellation requested")
        self.cancel_flag = True

        # Try to cancel the running task
        if self.current_task_id:
            cancel_task(self.client, self.current_task_id, self.logger)

        return True

    def _build_content(self, input_data: AppInput) -> list:
        """Build content list for BytePlus API."""
        content = []

        # Build text prompt with parameters
        text_content = build_text_content(
            input_data.prompt,
            duration=input_data.duration.value,
            camerafixed=str(input_data.camera_fixed).lower(),
        )
        content.append(text_content)

        # Add image if provided (image-to-video mode)
        if input_data.image:
            if not input_data.image.exists():
                raise RuntimeError(f"Input image does not exist at path: {input_data.image.path}")
            image_content = build_image_content(input_data.image.uri)
            content.append(image_content)

        return content

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Generate video using Seedance 1.5 Pro."""
        try:
            self.cancel_flag = False
            self.current_task_id = None

            mode = "image-to-video" if input_data.image else "text-to-video"
            self.logger.info(f"Starting {mode} generation")
            self.logger.info(f"Prompt: {input_data.prompt[:100]}...")
            self.logger.info(f"Duration: {input_data.duration.value}s, Camera fixed: {input_data.camera_fixed}")

            # Build content
            content = self._build_content(input_data)

            # Create task
            self.current_task_id = create_content_task(
                self.client,
                model=self.model_id,
                content=content,
                logger=self.logger,
            )

            # Poll for completion
            result = await poll_task_status(
                self.client,
                self.current_task_id,
                logger=self.logger,
                poll_interval=2.0,
                cancel_flag_getter=lambda: self.cancel_flag,
            )

            # Extract video URL from result
            # Note: Adjust based on actual response structure
            video_url = None
            if hasattr(result, 'data') and hasattr(result.data, 'video'):
                video_url = result.data.video.url
            elif hasattr(result, 'output') and isinstance(result.output, dict):
                video_url = result.output.get('video', {}).get('url')
            elif hasattr(result, 'video_url'):
                video_url = result.video_url

            if not video_url:
                self.logger.error(f"Could not extract video URL from result: {result}")
                raise RuntimeError("Failed to get video URL from response")

            # Download video
            video_path = download_video(video_url, self.logger)

            # Build output metadata for pricing
            duration_seconds = float(input_data.duration.value)
            output_meta = OutputMeta(
                outputs=[
                    VideoMeta(
                        width=1280,  # Default Seedance resolution
                        height=720,
                        resolution=VideoResolution.RES_720P,
                        seconds=duration_seconds,
                        fps=24,
                        extra={
                            "mode": mode,
                            "camera_fixed": input_data.camera_fixed,
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
        finally:
            self.current_task_id = None

"""
Seedance 1.0 Pro Fast - BytePlus Video Generation

Fast high-quality video generation from text prompts with optional first-frame image control.
Supports up to 1080p resolution with faster generation times. Uses BytePlus ARK SDK.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta, VideoResolution
from pydantic import Field
from typing import Optional
from enum import Enum
import logging

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


class ResolutionEnum(str, Enum):
    """Video resolution options."""
    p720 = "720p"
    p1080 = "1080p"


class AppInput(BaseAppInput):
    """Input schema for Seedance 1.0 Pro Fast video generation."""

    prompt: str = Field(
        description="Text prompt describing the video content and motion. Be descriptive about actions and camera movements.",
        examples=["At breakneck speed, drones thread through intricate obstacles, delivering an immersive flying experience."]
    )
    image: Optional[File] = Field(
        default=None,
        description="Optional first-frame image for image-to-video generation. If not provided, generates from text only."
    )
    resolution: ResolutionEnum = Field(
        default=ResolutionEnum.p1080,
        description="Video resolution. 1080p for higher quality, 720p for faster generation."
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
    """Output schema for Seedance 1.0 Pro Fast video generation."""

    video: File = Field(description="The generated video file.")


class App(BaseApp):
    """Seedance 1.0 Pro Fast video generation application using BytePlus ARK SDK."""

    async def setup(self, metadata):
        """Initialize the BytePlus client."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        logging.getLogger("httpx").setLevel(logging.WARNING)

        self.client = setup_byteplus_client()
        self.model_id = "seedance-1-0-pro-fast-251015"

        self.cancel_flag = False
        self.current_task_id = None

        self.logger.info(f"Seedance 1.0 Pro Fast initialized with model: {self.model_id}")

    async def on_cancel(self):
        """Handle cancellation request."""
        self.logger.info("Cancellation requested")
        self.cancel_flag = True
        if self.current_task_id:
            cancel_task(self.client, self.current_task_id, self.logger)
        return True

    def _build_content(self, input_data: AppInput) -> list:
        """Build content list for BytePlus API."""
        content = []

        text_content = build_text_content(
            input_data.prompt,
            resolution=input_data.resolution.value,
            duration=input_data.duration.value,
            camerafixed=str(input_data.camera_fixed).lower(),
        )
        content.append(text_content)

        if input_data.image:
            if not input_data.image.exists():
                raise RuntimeError(f"Input image does not exist at path: {input_data.image.path}")
            image_content = build_image_content(input_data.image.uri)
            content.append(image_content)

        return content

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Generate video using Seedance 1.0 Pro Fast."""
        try:
            self.cancel_flag = False
            self.current_task_id = None

            mode = "image-to-video" if input_data.image else "text-to-video"
            self.logger.info(f"Starting {mode} generation (fast mode)")
            self.logger.info(f"Prompt: {input_data.prompt[:100]}...")
            self.logger.info(f"Resolution: {input_data.resolution.value}, Duration: {input_data.duration.value}s")

            content = self._build_content(input_data)

            self.current_task_id = create_content_task(
                self.client,
                model=self.model_id,
                content=content,
                logger=self.logger,
            )

            result = await poll_task_status(
                self.client,
                self.current_task_id,
                logger=self.logger,
                poll_interval=2.0,
                cancel_flag_getter=lambda: self.cancel_flag,
            )

            video_url = None
            if hasattr(result, 'content') and hasattr(result.content, 'video_url'):
                video_url = result.content.video_url
            elif hasattr(result, 'video_url'):
                video_url = result.video_url

            if not video_url:
                self.logger.error(f"Could not extract video URL from result: {result}")
                raise RuntimeError("Failed to get video URL from response")

            video_path = download_video(video_url, self.logger)

            duration_seconds = getattr(result, 'duration', float(input_data.duration.value))
            fps = getattr(result, 'framespersecond', 24)
            resolution_str = getattr(result, 'resolution', input_data.resolution.value)
            seed = getattr(result, 'seed', None)

            # Extract token usage from response (for billing)
            usage = getattr(result, 'usage', None)
            completion_tokens = None
            total_tokens = None
            if usage:
                completion_tokens = getattr(usage, 'completion_tokens', None)
                total_tokens = getattr(usage, 'total_tokens', None)

            resolution_map = {
                '480p': VideoResolution.RES_480P,
                '720p': VideoResolution.RES_720P,
                '1080p': VideoResolution.RES_1080P,
            }
            resolution_enum = resolution_map.get(resolution_str, VideoResolution.RES_720P)

            width, height = (1280, 720)
            if resolution_str == '480p':
                width, height = (854, 480)
            elif resolution_str == '1080p':
                width, height = (1920, 1080)

            # Calculate estimated tokens as fallback
            estimated_tokens = int((width * height * fps * duration_seconds) / 1024)

            output_meta = OutputMeta(
                outputs=[
                    VideoMeta(
                        width=width,
                        height=height,
                        resolution=resolution_enum,
                        seconds=float(duration_seconds),
                        fps=fps,
                        extra={"mode": mode, "camera_fixed": input_data.camera_fixed, "seed": seed, "completion_tokens": completion_tokens, "total_tokens": total_tokens, "estimated_tokens": estimated_tokens}
                    )
                ]
            )

            self.logger.info(f"Video generated successfully: {video_path}")

            return AppOutput(video=File(path=video_path), output_meta=output_meta)

        except Exception as e:
            self.logger.error(f"Error during video generation: {e}")
            raise RuntimeError(f"Video generation failed: {str(e)}")
        finally:
            self.current_task_id = None

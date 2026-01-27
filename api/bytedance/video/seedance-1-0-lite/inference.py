"""
Seedance 1.0 Lite - BytePlus Video Generation

Lightweight video generation up to 1080p. Automatically selects:
- Image-to-video mode when an image is provided (supports watermark)
- Text-to-video mode when no image is provided (supports aspect ratio)

Uses BytePlus ARK SDK with async task polling.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta, VideoResolution
from pydantic import Field
from typing import Optional, ClassVar
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


class ResolutionEnum(str, Enum):
    """Video resolution options."""
    p480 = "480p"
    p720 = "720p"
    p1080 = "1080p"


class DurationEnum(str, Enum):
    """Video duration options in seconds (2-12s)."""
    s2 = "2"
    s3 = "3"
    s4 = "4"
    s5 = "5"
    s6 = "6"
    s7 = "7"
    s8 = "8"
    s9 = "9"
    s10 = "10"
    s11 = "11"
    s12 = "12"


class AspectRatioEnum(str, Enum):
    """Video aspect ratio options (text-to-video only)."""
    ratio_16_9 = "16:9"
    ratio_9_16 = "9:16"
    ratio_1_1 = "1:1"


class AppInput(BaseAppInput):
    """Input schema for Seedance 1.0 Lite video generation."""

    prompt: str = Field(
        description="Text prompt describing the video content and motion. Be descriptive about actions and camera movements.",
        examples=["Soft cotton-like clouds drift with subtle layered motions as the camera captures the scene."]
    )
    image: Optional[File] = Field(
        default=None,
        description="Optional first-frame image. If provided, uses image-to-video mode. If not, uses text-to-video mode."
    )
    resolution: ResolutionEnum = Field(
        default=ResolutionEnum.p720,
        description="Video resolution. 1080p for highest quality, 720p for balanced, 480p for fastest generation."
    )
    duration: DurationEnum = Field(
        default=DurationEnum.s5,
        description="Duration of the video in seconds (2-12 seconds)."
    )
    camera_fixed: bool = Field(
        default=False,
        description="Whether to fix the camera position during video generation. Set to true for static camera shots."
    )
    aspect_ratio: AspectRatioEnum = Field(
        default=AspectRatioEnum.ratio_16_9,
        description="Aspect ratio of the generated video (only applies to text-to-video mode when no image is provided)."
    )
    watermark: bool = Field(
        default=True,
        description="Whether to add a watermark to the generated video (only applies to image-to-video mode)."
    )


class AppOutput(BaseAppOutput):
    """Output schema for Seedance 1.0 Lite video generation."""

    video: File = Field(description="The generated video file.")


class App(BaseApp):
    """Seedance 1.0 Lite video generation application using BytePlus ARK SDK."""

    # Model IDs for each mode
    MODEL_I2V: ClassVar[str] = "seedance-1-0-lite-i2v-250428"
    MODEL_T2V: ClassVar[str] = "seedance-1-0-lite-t2v-250428"

    async def setup(self):
        """Initialize the BytePlus client."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        logging.getLogger("httpx").setLevel(logging.WARNING)

        self.client = setup_byteplus_client()

        self.cancel_flag = False
        self.current_task_id = None

        self.logger.info("Seedance 1.0 Lite initialized (supports both I2V and T2V)")

    async def on_cancel(self):
        """Handle cancellation request."""
        self.logger.info("Cancellation requested")
        self.cancel_flag = True
        if self.current_task_id:
            cancel_task(self.client, self.current_task_id, self.logger)
        return True

    def _get_dimensions(self, resolution: str, aspect_ratio: str) -> tuple:
        """Get video dimensions based on resolution and aspect ratio."""
        # Dimensions for each resolution and aspect ratio
        dimensions = {
            "480p": {
                "16:9": (854, 480),
                "9:16": (480, 854),
                "1:1": (480, 480),
            },
            "720p": {
                "16:9": (1280, 720),
                "9:16": (720, 1280),
                "1:1": (720, 720),
            },
            "1080p": {
                "16:9": (1920, 1080),
                "9:16": (1080, 1920),
                "1:1": (1080, 1080),
            },
        }
        return dimensions.get(resolution, dimensions["720p"]).get(aspect_ratio, (1280, 720))

    def _build_content(self, input_data: AppInput, is_i2v: bool) -> list:
        """Build content list for BytePlus API."""
        content = []

        if is_i2v:
            # Image-to-video mode
            text_content = build_text_content(
                input_data.prompt,
                resolution=input_data.resolution.value,
                duration=input_data.duration.value,
                camerafixed=str(input_data.camera_fixed).lower(),
                watermark=str(input_data.watermark).lower(),
            )
            content.append(text_content)

            if not input_data.image.exists():
                raise RuntimeError(f"Input image does not exist at path: {input_data.image.path}")
            image_content = build_image_content(input_data.image.uri)
            content.append(image_content)
        else:
            # Text-to-video mode
            text_content = build_text_content(
                input_data.prompt,
                ratio=input_data.aspect_ratio.value,
                resolution=input_data.resolution.value,
                duration=input_data.duration.value,
                camerafixed=str(input_data.camera_fixed).lower(),
            )
            content.append(text_content)

        return content

    async def run(self, input_data: AppInput) -> AppOutput:
        """Generate video using Seedance 1.0 Lite."""
        try:
            self.cancel_flag = False
            self.current_task_id = None

            # Determine mode based on image presence
            is_i2v = input_data.image is not None
            mode = "image-to-video" if is_i2v else "text-to-video"
            model_id = self.MODEL_I2V if is_i2v else self.MODEL_T2V

            self.logger.info(f"Starting {mode} generation (lite)")
            self.logger.info(f"Using model: {model_id}")
            self.logger.info(f"Prompt: {input_data.prompt[:100]}...")
            self.logger.info(f"Resolution: {input_data.resolution.value}, Duration: {input_data.duration.value}s")

            content = self._build_content(input_data, is_i2v)

            self.current_task_id = create_content_task(
                self.client,
                model=model_id,
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
            seed = getattr(result, 'seed', None)

            # Extract token usage from response (for billing)
            usage = getattr(result, 'usage', None)
            completion_tokens = None
            total_tokens = None
            if usage:
                completion_tokens = getattr(usage, 'completion_tokens', None)
                total_tokens = getattr(usage, 'total_tokens', None)

            # Get dimensions based on resolution and mode
            resolution_str = input_data.resolution.value
            if is_i2v:
                # For i2v, use 16:9 aspect ratio at the selected resolution
                width, height = self._get_dimensions(resolution_str, "16:9")
            else:
                width, height = self._get_dimensions(resolution_str, input_data.aspect_ratio.value)

            # Map resolution string to enum
            resolution_map = {
                '480p': VideoResolution.RES_480P,
                '720p': VideoResolution.RES_720P,
                '1080p': VideoResolution.RES_1080P,
            }
            resolution_enum = resolution_map.get(resolution_str, VideoResolution.RES_720P)

            # Calculate estimated tokens as fallback
            estimated_tokens = int((width * height * fps * duration_seconds) / 1024)

            # Build extra metadata based on mode
            extra = {
                "mode": mode,
                "camera_fixed": input_data.camera_fixed,
                "seed": seed,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "estimated_tokens": estimated_tokens,
            }
            if is_i2v:
                extra["watermark"] = input_data.watermark
            else:
                extra["aspect_ratio"] = input_data.aspect_ratio.value

            output_meta = OutputMeta(
                outputs=[
                    VideoMeta(
                        width=width,
                        height=height,
                        resolution=resolution_enum,
                        seconds=float(duration_seconds),
                        fps=fps,
                        extra=extra,
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

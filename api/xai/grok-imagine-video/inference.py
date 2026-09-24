"""
Grok Imagine Video - xAI Video Generation

Generate and edit videos using xAI's Grok Imagine Video model.
Supports text-to-video, image-to-video, and video editing.
"""

import asyncio
from typing import Optional, Literal
import tempfile

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta, VideoResolution, ImageMeta
from pydantic import Field
import requests

from .xai_helper import (
    VideoAspectRatioType,
    XAIError,
    MODERATION_NOTICE,
    is_moderated,
    upstream_cost_usd,
    create_xai_client,
    setup_logger,
    encode_image_base64,
    get_video_dimensions,
    retry_on_rate_limit,
    wait_for_video,
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

    video: Optional[File] = Field(default=None, description="The generated video file. Null when xAI content moderation withheld the video.")
    moderated: bool = Field(default=False, description="True when xAI content moderation withheld the generated video.")
    notice: Optional[str] = Field(default=None, description="Set when xAI content moderation withheld the video.")


class App(BaseApp):
    """Grok Imagine video generation application using xAI SDK."""

    async def setup(self):
        """Initialize the xAI client."""
        self.logger = setup_logger(__name__)
        self.client = create_xai_client()
        self.model = "grok-imagine-video"
        self.logger.info(f"Grok Imagine Video initialized with model: {self.model}")

    async def on_cancel(self):
        self._cancel = True
        return True

    def _cancelled(self) -> bool:
        ctx = getattr(self, "context", None)
        return self._cancel or bool(ctx is not None and getattr(ctx, "cancel_requested", False))

    async def run(self, input_data: AppInput) -> AppOutput:
        """Generate or edit video using Grok Imagine Video."""
        self._cancel = False
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

            # Start the job, then poll it without blocking the event loop: no
            # timeout (the platform owns it) and a cancel stops polling.
            start = await retry_on_rate_limit(
                lambda: self.client.video.start(**kwargs), logger=self.logger, in_thread=True,
            )
            self.logger.info(f"Video request started: {start.request_id}")
            response = await wait_for_video(self.client, start.request_id, self.logger, self._cancelled)

            # A moderated video is generated and billed by xAI, so the run
            # succeeds and is billed with the video left out.
            moderated = is_moderated(response)
            cost_usd = upstream_cost_usd(response)
            video_file = None
            if moderated:
                self.logger.warning("xAI content moderation withheld the video")
            else:
                video_url = response.url
                if not video_url:
                    raise RuntimeError("No video URL in response")

                self.logger.info(f"Downloading video from: {video_url}")
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                    video_response = await asyncio.to_thread(requests.get, video_url, timeout=300)
                    video_response.raise_for_status()
                    f.write(video_response.content)
                    video_path = f.name
                video_file = File(path=video_path)

            # Get duration from response or use input
            duration_seconds = getattr(response, 'duration', float(input_data.duration))

            # Map resolution to enum and get dimensions
            resolution_enum_map = {
                "480p": VideoResolution.VIDEO_RES480_P,
                "720p": VideoResolution.VIDEO_RES720_P,
            }
            resolution_enum = resolution_enum_map.get(input_data.resolution, VideoResolution.VIDEO_RES720_P)
            width, height = get_video_dimensions(input_data.aspect_ratio, input_data.resolution)

            # Track inputs for billing
            input_metas = []
            if input_data.image:
                # Image input: $0.002
                input_metas.append(ImageMeta(width=0, height=0, count=1, extra={"type": "image_input"}))
            if input_data.video:
                # Video input: $0.01/s — probe duration
                input_video_seconds = 0.0
                if input_data.video.exists():
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
                input_metas.append(VideoMeta(
                    width=0, height=0, resolution=VideoResolution.VIDEO_RES480_P,
                    seconds=input_video_seconds, fps=24, extra={"type": "video_input"}
                ))

            output_meta = OutputMeta(
                inputs=input_metas,
                outputs=[
                    VideoMeta(
                        width=width,
                        height=height,
                        resolution=resolution_enum,
                        seconds=float(duration_seconds),
                        fps=24,
                        extra={
                            "mode": mode,
                            "upstream_cost_usd": cost_usd,
                            "aspect_ratio": input_data.aspect_ratio,
                        }
                    )
                ]
            )

            self.logger.info(f"Video generated: moderated={moderated}, billed {float(duration_seconds)}s output, xAI cost ${cost_usd}")

            return AppOutput(
                video=video_file,
                moderated=moderated,
                notice=MODERATION_NOTICE.format(what="video") if moderated else None,
                output_meta=output_meta,
            )

        except XAIError:
            raise
        except Exception as e:
            self.logger.error(f"Error during video generation: {e}")
            raise RuntimeError(f"Video generation failed: {str(e)}")

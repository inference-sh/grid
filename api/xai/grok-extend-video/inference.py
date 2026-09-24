"""
Grok Extend Video - xAI Video Extension

Extend existing videos using xAI's Grok Imagine Video model.
Takes an existing video and generates additional frames to continue it.
"""

import asyncio
from typing import Optional, Literal

import tempfile

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta, VideoResolution, ImageMeta
from pydantic import Field
import requests

from .xai_helper import (
    XAIError,
    MODERATION_NOTICE,
    is_moderated,
    upstream_cost_usd,
    create_xai_client,
    setup_logger,
    retry_on_rate_limit,
    wait_for_video,
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

    video: Optional[File] = Field(default=None, description="The extended video file. Null when xAI content moderation withheld the video.")
    moderated: bool = Field(default=False, description="True when xAI content moderation withheld the generated video.")
    notice: Optional[str] = Field(default=None, description="Set when xAI content moderation withheld the video.")


class App(BaseApp):
    """Grok video extension application using xAI SDK."""

    async def setup(self):
        """Initialize the xAI client."""
        self.logger = setup_logger(__name__)
        self.client = create_xai_client()
        self.model = "grok-imagine-video"
        self.logger.info(f"Grok Extend Video initialized with model: {self.model}")

    async def on_cancel(self):
        self._cancel = True
        return True

    def _cancelled(self) -> bool:
        ctx = getattr(self, "context", None)
        return self._cancel or bool(ctx is not None and getattr(ctx, "cancel_requested", False))

    async def run(self, input_data: AppInput) -> AppOutput:
        """Extend a video using Grok Imagine Video."""
        self._cancel = False
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

            # Start the job, then poll it without blocking the event loop: no
            # timeout (the platform owns it) and a cancel stops polling.
            start = await retry_on_rate_limit(
                lambda: self.client.video.extend_start(**kwargs), logger=self.logger, in_thread=True,
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

                self.logger.info(f"Downloading extended video from: {video_url}")
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                    video_response = await asyncio.to_thread(requests.get, video_url, timeout=300)
                    video_response.raise_for_status()
                    f.write(video_response.content)
                    video_path = f.name
                video_file = File(path=video_path)

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
                            "upstream_cost_usd": cost_usd,
                        }
                    )
                ]
            )

            self.logger.info(f"Video extended: moderated={moderated}, billed {float(duration_seconds)}s output, xAI cost ${cost_usd}")

            return AppOutput(
                video=video_file,
                moderated=moderated,
                notice=MODERATION_NOTICE.format(what="video") if moderated else None,
                output_meta=output_meta,
            )

        except XAIError:
            raise
        except Exception as e:
            self.logger.error(f"Error during video extension: {e}")
            raise RuntimeError(f"Video extension failed: {str(e)}")

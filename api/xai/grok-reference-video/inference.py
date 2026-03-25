"""
Grok Reference Video - xAI Reference-to-Video Generation

Generate videos using reference images for style and content guidance.
Uses xAI's Grok Imagine Video model with reference_image_urls.
"""

from typing import Optional, Literal, List

import tempfile

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta, VideoResolution, ImageMeta
from pydantic import Field
import requests

from .xai_helper import (
    VideoAspectRatioType,
    XAIError,
    create_xai_client,
    setup_logger,
    encode_image_base64,
    get_video_dimensions,
    retry_on_rate_limit,
)


ResolutionType = Literal["720p", "480p"]


class AppInput(BaseAppInput):
    """Input schema for Grok reference-to-video generation."""

    prompt: str = Field(
        description="Text prompt describing the video content and motion.",
        examples=["A person walking through a city in the style of the reference images"]
    )
    reference_images: List[File] = Field(
        description="Reference images for style/content guidance. The generated video will use these images as visual references.",
        min_length=1,
    )
    duration: int = Field(
        default=5,
        ge=1,
        le=15,
        description="Duration of the generated video in seconds (1-15)."
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
    """Output schema for Grok reference-to-video generation."""

    video: File = Field(description="The generated video file.")


class App(BaseApp):
    """Grok reference-to-video generation application using xAI SDK."""

    async def setup(self):
        """Initialize the xAI client."""
        self.logger = setup_logger(__name__)
        self.client = create_xai_client()
        self.model = "grok-imagine-video"
        self.logger.info(f"Grok Reference Video initialized with model: {self.model}")

    async def run(self, input_data: AppInput) -> AppOutput:
        """Generate video using reference images with Grok Imagine Video."""
        try:
            self.logger.info(f"Starting reference-to-video generation")
            self.logger.info(f"Prompt: {input_data.prompt[:100]}...")
            self.logger.info(f"Reference images: {len(input_data.reference_images)}")
            self.logger.info(f"Duration: {input_data.duration}s, Resolution: {input_data.resolution}, Aspect ratio: {input_data.aspect_ratio}")

            # Build reference image URLs
            reference_image_urls = []
            for img in input_data.reference_images:
                if img.uri and img.uri.startswith("http"):
                    reference_image_urls.append(img.uri)
                else:
                    reference_image_urls.append(encode_image_base64(img))

            kwargs = {
                "model": self.model,
                "prompt": input_data.prompt,
                "duration": input_data.duration,
                "aspect_ratio": input_data.aspect_ratio,
                "resolution": input_data.resolution,
                "reference_image_urls": reference_image_urls,
            }

            # Generate video (SDK handles polling automatically, with 429 retry)
            self.logger.info("Starting reference-to-video generation (SDK will poll automatically)...")
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
                "480p": VideoResolution.VIDEO_RES480_P,
                "720p": VideoResolution.VIDEO_RES720_P,
            }
            resolution_enum = resolution_enum_map.get(input_data.resolution, VideoResolution.VIDEO_RES720_P)
            width, height = get_video_dimensions(input_data.aspect_ratio, input_data.resolution)

            # Track input images for billing (image input: $0.002 each)
            input_metas = [
                ImageMeta(
                    width=0,
                    height=0,
                    count=1,
                    extra={"type": "reference_image"}
                )
                for _ in input_data.reference_images
            ]

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
                            "mode": "reference-to-video",
                            "aspect_ratio": input_data.aspect_ratio,
                            "reference_count": len(input_data.reference_images),
                        }
                    )
                ]
            )

            self.logger.info(f"Video generated successfully: {video_path}")

            return AppOutput(
                video=File(path=video_path),
                output_meta=output_meta,
            )

        except XAIError:
            raise
        except Exception as e:
            self.logger.error(f"Error during reference-to-video generation: {e}")
            raise RuntimeError(f"Reference-to-video generation failed: {str(e)}")

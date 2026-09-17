"""
WAN-I2V - Image-to-video generation by Pruna
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta, VideoMeta, VideoResolution
from pydantic import Field
from typing import Optional
from enum import Enum
import logging

from .pruna_helper import run_prediction, get_generation_url, download_video, upload_file, probe_media_duration

# Upstream wan-i2v takes no duration parameter; clips are a fixed ~5 seconds
FIXED_CLIP_SECONDS = 5.0


class ResolutionEnum(str, Enum):
    sd = "480p"
    hd = "720p"


class AppInput(BaseAppInput):
    """Input schema for WAN-I2V."""

    prompt: str = Field(
        description="Text description for video generation."
    )
    image: File = Field(
        description="Input image for image-to-video generation."
    )
    resolution: ResolutionEnum = Field(
        default=ResolutionEnum.sd,
        description="Video resolution: 480p or 720p."
    )
    duration: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Ignored. The upstream model does not accept a duration and always returns a fixed-length clip (about 5 seconds). Billing uses the actual length of the returned video."
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducible generation."
    )


class AppOutput(BaseAppOutput):
    """Output schema for WAN-I2V."""

    video: File = Field(description="Generated video file.")
    seed: Optional[int] = Field(default=None, description="Seed used for generation.")


class App(BaseApp):
    """WAN-I2V image-to-video generation."""

    async def setup(self):
        """Initialize the application."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.model = "wan-i2v"
        self.logger.info("WAN-I2V initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        """Generate video using WAN-I2V."""
        try:
            self.logger.info(f"Generating video from image: {input_data.prompt[:100]}...")
            self.logger.info(f"Resolution: {input_data.resolution.value}")
            if input_data.duration != 5:
                self.logger.warning(f"duration={input_data.duration} is ignored: upstream returns a fixed-length clip")

            # Handle image input
            if not input_data.image.exists():
                raise RuntimeError(f"Input image does not exist: {input_data.image.path}")

            if input_data.image.uri and input_data.image.uri.startswith("http"):
                image_url = input_data.image.uri
            else:
                self.logger.info("Uploading input image...")
                upload_result = upload_file(input_data.image.path, logger=self.logger)
                image_url = upload_result.get("urls", {}).get("get")
                if not image_url:
                    raise RuntimeError("Failed to get URL for uploaded image")

            # Build request
            request_data = {
                "prompt": input_data.prompt,
                "image": image_url,
                "resolution": input_data.resolution.value,
            }

            if input_data.seed is not None:
                request_data["seed"] = input_data.seed

            # Video generation needs async polling
            result = await run_prediction(
                model=self.model,
                input_data=request_data,
                use_sync=False,
                logger=self.logger,
            )

            # Download result
            generation_url = get_generation_url(result)

            video_path = download_video(generation_url, logger=self.logger)

            # Bill the delivered length, never the requested duration (upstream ignores it)
            video_seconds = float(result.get("duration") or 0) or probe_media_duration(video_path, logger=self.logger)
            if video_seconds <= 0:
                self.logger.warning(f"Could not determine output duration, billing fixed clip length {FIXED_CLIP_SECONDS}s")
                video_seconds = FIXED_CLIP_SECONDS

            # Read input image dimensions
            from PIL import Image as PILImage
            with PILImage.open(input_data.image.path) as pil_img:
                in_w, in_h = pil_img.size

            # Build output metadata for pricing
            resolution_map = {
                "480p": VideoResolution.VIDEO_RES480_P,
                "720p": VideoResolution.VIDEO_RES720_P,
            }

            dims_map = {
                "480p": (854, 480),
                "720p": (1280, 720),
            }

            width, height = dims_map.get(input_data.resolution.value, (854, 480))

            output_meta = OutputMeta(
                inputs=[ImageMeta(width=in_w, height=in_h, count=1)],
                outputs=[
                    VideoMeta(
                        width=width,
                        height=height,
                        resolution=resolution_map.get(input_data.resolution.value, VideoResolution.VIDEO_RES480_P),
                        seconds=video_seconds,
                    )
                ],
            )

            self.logger.info("Video generated successfully")

            return AppOutput(
                video=File(path=video_path),
                seed=result.get("seed"),
                output_meta=output_meta,
            )

        except Exception as e:
            self.logger.error(f"Error: {e}")
            raise RuntimeError(f"Video generation failed: {str(e)}")

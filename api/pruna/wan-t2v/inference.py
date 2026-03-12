"""
WAN-T2V - Text-to-video generation by Pruna
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta, VideoResolution
from pydantic import Field
from typing import Optional
from enum import Enum
import logging

from .pruna_helper import run_prediction, download_video


class ResolutionEnum(str, Enum):
    sd = "480p"
    hd = "720p"


class AspectRatioEnum(str, Enum):
    landscape = "16:9"
    portrait = "9:16"
    square = "1:1"


class AppInput(BaseAppInput):
    """Input schema for WAN-T2V."""

    prompt: str = Field(
        description="Text description for video generation."
    )
    resolution: ResolutionEnum = Field(
        default=ResolutionEnum.sd,
        description="Video resolution: 480p or 720p."
    )
    aspect_ratio: AspectRatioEnum = Field(
        default=AspectRatioEnum.landscape,
        description="Aspect ratio for the video."
    )
    duration: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Video duration in seconds (1-10)."
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducible generation."
    )


class AppOutput(BaseAppOutput):
    """Output schema for WAN-T2V."""

    video: File = Field(description="Generated video file.")
    seed: Optional[int] = Field(default=None, description="Seed used for generation.")


class App(BaseApp):
    """WAN-T2V text-to-video generation."""

    async def setup(self, metadata):
        """Initialize the application."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.model = "wan-t2v"
        self.logger.info("WAN-T2V initialized")

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Generate video using WAN-T2V."""
        try:
            self.logger.info(f"Generating video: {input_data.prompt[:100]}...")
            self.logger.info(f"Resolution: {input_data.resolution.value}, Duration: {input_data.duration}s")

            # Build request
            request_data = {
                "prompt": input_data.prompt,
                "resolution": input_data.resolution.value,
                "aspect_ratio": input_data.aspect_ratio.value,
                "duration": input_data.duration,
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
            generation_url = result.get("generation_url")
            if not generation_url:
                raise RuntimeError("No generation_url in response")

            if generation_url.startswith("/"):
                generation_url = f"https://api.pruna.ai{generation_url}"

            video_path = download_video(generation_url, logger=self.logger)

            # Build output metadata for pricing
            resolution_map = {
                "480p": VideoResolution.VIDEO_RES480_P,
                "720p": VideoResolution.VIDEO_RES720_P,
            }

            dims_map = {
                ("480p", "16:9"): (854, 480),
                ("480p", "9:16"): (480, 854),
                ("480p", "1:1"): (480, 480),
                ("720p", "16:9"): (1280, 720),
                ("720p", "9:16"): (720, 1280),
                ("720p", "1:1"): (720, 720),
            }

            width, height = dims_map.get(
                (input_data.resolution.value, input_data.aspect_ratio.value),
                (854, 480)
            )

            output_meta = OutputMeta(
                outputs=[
                    VideoMeta(
                        width=width,
                        height=height,
                        resolution=resolution_map.get(input_data.resolution.value, VideoResolution.VIDEO_RES480_P),
                        seconds=float(input_data.duration),
                    )
                ]
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

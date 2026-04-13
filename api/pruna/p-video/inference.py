"""
P-Video - Premium AI video generation from text, images, and audio by Pruna
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta, VideoMeta, VideoResolution
from pydantic import Field
from typing import Optional
from enum import Enum
import logging

from .pruna_helper import run_prediction, get_generation_url, download_video, upload_file


class ResolutionEnum(str, Enum):
    hd = "720p"
    full_hd = "1080p"


class AspectRatioEnum(str, Enum):
    landscape = "16:9"
    portrait = "9:16"
    photo_landscape = "4:3"
    photo_portrait = "3:4"
    classic_landscape = "3:2"
    classic_portrait = "2:3"
    square = "1:1"


class FpsEnum(int, Enum):
    standard = 24
    high = 48


class AppInput(BaseAppInput):
    """Input schema for P-Video."""

    prompt: str = Field(
        description="Text description for video generation."
    )
    image: Optional[File] = Field(
        default=None,
        description="Input image for image-to-video. When provided, aspect_ratio is ignored."
    )
    audio: Optional[File] = Field(
        default=None,
        description="Audio file for audio-conditioned video. When provided, duration is ignored. Supports flac, mp3, wav."
    )
    duration: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Video duration in seconds (1-10). Ignored if audio is provided."
    )
    resolution: ResolutionEnum = Field(
        default=ResolutionEnum.hd,
        description="Video resolution: 720p or 1080p."
    )
    fps: FpsEnum = Field(
        default=FpsEnum.standard,
        description="Frames per second: 24 or 48."
    )
    aspect_ratio: AspectRatioEnum = Field(
        default=AspectRatioEnum.landscape,
        description="Aspect ratio. Ignored when input image is provided."
    )
    draft: bool = Field(
        default=False,
        description="Draft mode for cheaper, lower-quality previews."
    )
    save_audio: bool = Field(
        default=True,
        description="Include audio in output video."
    )
    prompt_upsampling: bool = Field(
        default=True,
        description="Enhance prompt with LLM."
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducible generation."
    )
    disable_safety_filter: bool = Field(
        default=True,
        description="Disable safety filter for prompts and input images."
    )


class AppOutput(BaseAppOutput):
    """Output schema for P-Video."""

    video: File = Field(description="Generated video file.")
    seed: Optional[int] = Field(default=None, description="Seed used for generation.")


class App(BaseApp):
    """P-Video for AI video generation."""

    async def setup(self):
        """Initialize the application."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.model = "p-video"
        self.logger.info("P-Video initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        """Generate video using P-Video."""
        try:
            self.logger.info(f"Generating video: {input_data.prompt[:100]}...")
            self.logger.info(f"Resolution: {input_data.resolution.value}, Duration: {input_data.duration}s")

            # Build request
            request_data = {
                "prompt": input_data.prompt,
                "duration": input_data.duration,
                "resolution": input_data.resolution.value,
                "fps": input_data.fps.value,
                "aspect_ratio": input_data.aspect_ratio.value,
                "draft": input_data.draft,
                "save_audio": input_data.save_audio,
                "prompt_upsampling": input_data.prompt_upsampling,
                "disable_safety_filter": input_data.disable_safety_filter,
            }

            # Handle image input (i2v)
            if input_data.image:
                if not input_data.image.exists():
                    raise RuntimeError(f"Input image does not exist: {input_data.image.path}")

                if input_data.image.uri and input_data.image.uri.startswith("http"):
                    request_data["image"] = input_data.image.uri
                else:
                    self.logger.info("Uploading input image...")
                    upload_result = upload_file(input_data.image.path, logger=self.logger)
                    image_url = upload_result.get("urls", {}).get("get")
                    if not image_url:
                        raise RuntimeError("Failed to get URL for uploaded image")
                    request_data["image"] = image_url

            # Handle audio input
            if input_data.audio:
                if not input_data.audio.exists():
                    raise RuntimeError(f"Audio file does not exist: {input_data.audio.path}")

                if input_data.audio.uri and input_data.audio.uri.startswith("http"):
                    request_data["audio"] = input_data.audio.uri
                else:
                    self.logger.info("Uploading audio file...")
                    upload_result = upload_file(input_data.audio.path, logger=self.logger)
                    audio_url = upload_result.get("urls", {}).get("get")
                    if not audio_url:
                        raise RuntimeError("Failed to get URL for uploaded audio")
                    request_data["audio"] = audio_url

            if input_data.seed is not None:
                request_data["seed"] = input_data.seed

            # Video generation is slower - use async polling (not sync mode)
            result = await run_prediction(
                model=self.model,
                input_data=request_data,
                use_sync=False,  # Video needs async polling
                logger=self.logger,
            )

            # Download result
            generation_url = get_generation_url(result)

            video_path = download_video(generation_url, logger=self.logger)

            # Build output metadata for pricing
            resolution_map = {
                "720p": VideoResolution.VIDEO_RES720_P,
                "1080p": VideoResolution.VIDEO_RES1080_P,
            }

            # Dimensions based on resolution and aspect ratio
            dims_map = {
                ("720p", "16:9"): (1280, 720),
                ("720p", "9:16"): (720, 1280),
                ("720p", "1:1"): (720, 720),
                ("720p", "4:3"): (960, 720),
                ("720p", "3:4"): (720, 960),
                ("720p", "3:2"): (1080, 720),
                ("720p", "2:3"): (720, 1080),
                ("1080p", "16:9"): (1920, 1080),
                ("1080p", "9:16"): (1080, 1920),
                ("1080p", "1:1"): (1080, 1080),
                ("1080p", "4:3"): (1440, 1080),
                ("1080p", "3:4"): (1080, 1440),
                ("1080p", "3:2"): (1620, 1080),
                ("1080p", "2:3"): (1080, 1620),
            }

            width, height = dims_map.get(
                (input_data.resolution.value, input_data.aspect_ratio.value),
                (1280, 720)
            )

            # Read input file dimensions if provided
            input_metas = []
            if input_data.image and input_data.image.path:
                from PIL import Image
                with Image.open(input_data.image.path) as pil_img:
                    in_w, in_h = pil_img.size
                input_metas.append(ImageMeta(width=in_w, height=in_h, count=1))

            output_meta = OutputMeta(
                inputs=input_metas,
                outputs=[
                    VideoMeta(
                        width=width,
                        height=height,
                        resolution=resolution_map.get(input_data.resolution.value, VideoResolution.VIDEO_RES720_P),
                        seconds=float(input_data.duration),
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

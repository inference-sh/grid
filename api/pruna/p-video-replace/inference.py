"""
P-Video-Replace - Replace characters in videos using reference images by Pruna

Takes a source video and 1-4 reference images, replaces characters while
preserving motion, timing, camera movement, and scene structure.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta, VideoResolution
from pydantic import Field
from typing import Optional, List
from enum import Enum
import logging

from .pruna_helper import run_prediction, get_generation_url, download_video, upload_file


class ResolutionEnum(str, Enum):
    hd = "720p"
    full_hd = "1080p"


class FpsEnum(str, Enum):
    original = "original"
    fps_24 = "24"
    fps_48 = "48"


class AppInput(BaseAppInput):
    """Input schema for P-Video-Replace."""

    video: File = Field(
        description="Source video (.mp4). Motion, timing, camera, and scene come from this."
    )
    images: List[File] = Field(
        description="Reference image(s) of people to place into the video (1-4 images).",
        min_length=1,
        max_length=4,
    )
    instruction_prompt: Optional[str] = Field(
        default=None,
        description="Describe who in the video to replace with whom from reference images. E.g. 'Replace the person in the source video with the woman from reference image 1. Keep lip sync, motion, audio, and camera from the source video.'",
    )
    resolution: ResolutionEnum = Field(
        default=ResolutionEnum.hd,
        description="Output resolution: 720p ($0.03/s) or 1080p ($0.06/s)."
    )
    turbo: bool = Field(
        default=False,
        description="Turbo mode: faster generation for slightly lower quality."
    )
    target_fps: FpsEnum = Field(
        default=FpsEnum.original,
        description="Target FPS: 'original' to match source, or '24'/'48'."
    )
    save_audio: bool = Field(
        default=True,
        description="Include audio from the source video in the output."
    )
    ignore_audio: bool = Field(
        default=False,
        description="Ignore source audio during generation. If save_audio is true, audio is still saved."
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducible generation."
    )
    disable_safety_checker: bool = Field(
        default=False,
        description="Disable safety checker for generated videos."
    )


class AppOutput(BaseAppOutput):
    """Output schema for P-Video-Replace."""

    video: File = Field(description="Generated video with replaced characters.")
    seed: Optional[int] = Field(default=None, description="Seed used for generation.")


class App(BaseApp):
    """P-Video-Replace for character replacement in videos."""

    async def setup(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.model = "p-video-replace"
        self.logger.info("P-Video-Replace initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        """Replace characters in video using reference images."""
        self.logger.info(f"Replacing characters, resolution: {input_data.resolution.value}")
        self.logger.info(f"Reference images: {len(input_data.images)}")

        # Upload source video
        if not input_data.video.exists():
            raise RuntimeError(f"Source video does not exist: {input_data.video.path}")

        if input_data.video.uri and input_data.video.uri.startswith("http"):
            video_url = input_data.video.uri
        else:
            self.logger.info("Uploading source video...")
            upload_result = upload_file(input_data.video.path, logger=self.logger)
            video_url = upload_result.get("urls", {}).get("get")
            if not video_url:
                raise RuntimeError("Failed to get URL for uploaded video")

        # Upload reference images
        image_urls = []
        for i, img in enumerate(input_data.images):
            if not img.exists():
                raise RuntimeError(f"Reference image {i+1} does not exist: {img.path}")

            if img.uri and img.uri.startswith("http"):
                image_urls.append(img.uri)
            else:
                self.logger.info(f"Uploading reference image {i+1}...")
                upload_result = upload_file(img.path, logger=self.logger)
                img_url = upload_result.get("urls", {}).get("get")
                if not img_url:
                    raise RuntimeError(f"Failed to get URL for reference image {i+1}")
                image_urls.append(img_url)

        # Build request
        request_data = {
            "video": video_url,
            "images": image_urls,
            "resolution": input_data.resolution.value,
            "turbo": input_data.turbo,
            "save_audio": input_data.save_audio,
            "ignore_audio": input_data.ignore_audio,
            "disable_safety_checker": input_data.disable_safety_checker,
        }

        if input_data.target_fps != FpsEnum.original:
            request_data["target_fps"] = input_data.target_fps.value

        if input_data.instruction_prompt:
            request_data["instruction_prompt"] = input_data.instruction_prompt
            self.logger.info(f"Instruction: {input_data.instruction_prompt[:100]}...")

        if input_data.seed is not None:
            request_data["seed"] = input_data.seed

        # Run prediction (async polling — video generation is slow)
        result = await run_prediction(
            model=self.model,
            input_data=request_data,
            use_sync=False,
            logger=self.logger,
        )

        # Download result
        generation_url = get_generation_url(result)
        video_path = download_video(generation_url, logger=self.logger)

        # Get video duration from result or probe
        video_seconds = float(result.get("duration", 0))
        if video_seconds == 0:
            # Try to get duration with ffprobe
            import subprocess
            import json as _json
            try:
                probe = subprocess.run(
                    ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", video_path],
                    capture_output=True, text=True, timeout=30,
                )
                if probe.returncode == 0:
                    fmt = _json.loads(probe.stdout).get("format", {})
                    video_seconds = float(fmt.get("duration", 0))
            except Exception:
                pass

        # Output metadata for pricing
        resolution_map = {
            "720p": VideoResolution.VIDEO_RES720_P,
            "1080p": VideoResolution.VIDEO_RES1080_P,
        }

        # Aspect ratio follows the source video, use reasonable defaults
        dims_map = {
            "720p": (1280, 720),
            "1080p": (1920, 1080),
        }
        width, height = dims_map.get(input_data.resolution.value, (1280, 720))

        output_meta = OutputMeta(
            outputs=[
                VideoMeta(
                    width=width,
                    height=height,
                    resolution=resolution_map.get(input_data.resolution.value, VideoResolution.VIDEO_RES720_P),
                    seconds=video_seconds,
                    extra={"resolution": input_data.resolution.value},
                )
            ],
        )

        self.logger.info(f"Video replaced: {video_seconds:.1f}s at {input_data.resolution.value}")

        return AppOutput(
            video=File(path=video_path),
            seed=result.get("seed"),
            output_meta=output_meta,
        )

"""
Kling O3 Video-to-Video

Transform videos using Kling O3 model via fal.ai. Supports edit and reference
modes, with pro and standard tiers.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta
from pydantic import Field
from typing import List, Optional
from enum import Enum
import logging

from .fal_helper import setup_fal_client, run_fal_model, download_video

logging.getLogger("httpx").setLevel(logging.WARNING)

VIDEO_DIMENSIONS = {
    "16:9": (1280, 720),
    "9:16": (720, 1280),
    "1:1": (720, 720),
}


class TierEnum(str, Enum):
    pro = "pro"
    standard = "standard"


class ModeEnum(str, Enum):
    edit = "edit"
    reference = "reference"


class AspectRatioEnum(str, Enum):
    ratio_16_9 = "16:9"
    ratio_9_16 = "9:16"
    ratio_1_1 = "1:1"


class DurationEnum(str, Enum):
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
    s13 = "13"
    s14 = "14"
    s15 = "15"


class AppInput(BaseAppInput):
    """Input schema for Kling O3 Video-to-Video."""

    prompt: str = Field(
        description="The text prompt describing the desired transformation.",
        examples=["Transform the scene into a watercolor painting style"],
    )
    video: File = Field(
        description="Input video to transform. Supported formats: MP4, MOV.",
    )
    mode: ModeEnum = Field(
        default=ModeEnum.edit,
        description="Video-to-video mode. 'edit' modifies the input video, 'reference' uses it as style reference.",
    )
    tier: TierEnum = Field(
        default=TierEnum.pro,
        description="Model tier. Pro offers higher quality, standard is faster and cheaper.",
    )
    image_urls: Optional[List[File]] = Field(
        default=None,
        description="Optional reference images for style guidance.",
    )
    keep_audio: bool = Field(
        default=True,
        description="Whether to keep the original audio from the input video.",
    )
    aspect_ratio: Optional[AspectRatioEnum] = Field(
        default=None,
        description="Aspect ratio for the output video. Only used in reference mode.",
    )
    duration: Optional[DurationEnum] = Field(
        default=None,
        description="Duration of the output video in seconds (3-15). Only used in reference mode.",
    )


class AppOutput(BaseAppOutput):
    """Output schema for Kling O3 Video-to-Video."""

    video: File = Field(description="The transformed video file.")


class App(BaseApp):
    """Kling O3 Video-to-Video application."""

    async def setup(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Kling O3 V2V initialized")

    def _build_request(self, input_data: AppInput) -> dict:
        request = {
            "prompt": input_data.prompt,
            "video_url": input_data.video.uri,
            "keep_audio": input_data.keep_audio,
        }
        if input_data.image_urls:
            request["image_urls"] = [f.uri for f in input_data.image_urls]
        # Reference mode supports aspect_ratio and duration
        if input_data.mode == ModeEnum.reference:
            if input_data.aspect_ratio:
                request["aspect_ratio"] = input_data.aspect_ratio.value
            if input_data.duration:
                request["duration"] = input_data.duration.value
        return request

    async def run(self, input_data: AppInput) -> AppOutput:
        try:
            if not input_data.video.exists():
                raise RuntimeError(f"Input video does not exist at path: {input_data.video.path}")
            if input_data.image_urls:
                for idx, f in enumerate(input_data.image_urls):
                    if f and not f.exists():
                        raise RuntimeError(f"Reference image {idx + 1} does not exist at path: {f.path}")

            setup_fal_client()

            model_id = f"fal-ai/kling-video/o3/{input_data.tier.value}/video-to-video/{input_data.mode.value}"
            self.logger.info(
                f"Transforming video ({input_data.tier.value}/{input_data.mode.value}) "
                f"with prompt: {input_data.prompt[:100]}..."
            )
            self.logger.info(f"Settings: keep_audio={input_data.keep_audio}, endpoint={model_id}")

            request_data = self._build_request(input_data)
            result = run_fal_model(model_id, request_data, self.logger)

            video_url = result["video"]["url"]
            video_path = download_video(video_url, self.logger)

            # Determine dimensions from aspect_ratio if in reference mode, else default
            if input_data.mode == ModeEnum.reference and input_data.aspect_ratio:
                width, height = VIDEO_DIMENSIONS.get(input_data.aspect_ratio.value, (1280, 720))
            else:
                width, height = 1280, 720

            seconds = float(input_data.duration.value) if input_data.duration else 5.0

            output_meta = OutputMeta(
                outputs=[
                    VideoMeta(
                        width=width,
                        height=height,
                        seconds=seconds,
                        fps=24,
                    )
                ]
            )

            return AppOutput(
                video=File(path=video_path),
                output_meta=output_meta,
            )

        except Exception as e:
            self.logger.error(f"Error during video transformation: {e}")
            raise RuntimeError(f"Video transformation failed: {str(e)}")

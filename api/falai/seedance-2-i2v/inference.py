"""
Seedance 2.0 Image-to-Video

Generate videos with synchronized audio from a starting image (with optional
end frame) using ByteDance's Seedance 2.0 model. Supports quality and fast modes.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta, VideoResolution
from pydantic import Field
from typing import Optional
from enum import Enum
import logging

from .fal_helper import setup_fal_client, run_fal_model, download_video

logging.getLogger("httpx").setLevel(logging.WARNING)


class ModeEnum(str, Enum):
    """Generation mode — quality uses the full model, fast is ~20% cheaper and quicker."""
    quality = "quality"
    fast = "fast"


class AspectRatioEnum(str, Enum):
    """Video aspect ratio options."""
    auto = "auto"
    ratio_21_9 = "21:9"
    ratio_16_9 = "16:9"
    ratio_4_3 = "4:3"
    ratio_1_1 = "1:1"
    ratio_3_4 = "3:4"
    ratio_9_16 = "9:16"


class ResolutionEnum(str, Enum):
    """Video resolution options."""
    p480 = "480p"
    p720 = "720p"


class DurationEnum(str, Enum):
    """Video duration options in seconds."""
    auto = "auto"
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


VIDEO_DIMENSIONS = {
    ("480p", "21:9"): (1120, 480),
    ("480p", "16:9"): (854, 480),
    ("480p", "4:3"): (640, 480),
    ("480p", "1:1"): (480, 480),
    ("480p", "3:4"): (480, 640),
    ("480p", "9:16"): (480, 854),
    ("480p", "auto"): (854, 480),
    ("720p", "21:9"): (1680, 720),
    ("720p", "16:9"): (1280, 720),
    ("720p", "4:3"): (960, 720),
    ("720p", "1:1"): (720, 720),
    ("720p", "3:4"): (720, 960),
    ("720p", "9:16"): (720, 1280),
    ("720p", "auto"): (1280, 720),
}

AUTO_DURATION_SECONDS = 5.0

ENDPOINT_BY_MODE = {
    ModeEnum.quality: "bytedance/seedance-2.0/image-to-video",
    ModeEnum.fast: "bytedance/seedance-2.0/fast/image-to-video",
}


class AppInput(BaseAppInput):
    """Input schema for Seedance 2.0 Image-to-Video."""

    prompt: str = Field(
        description="The text prompt describing the desired motion and action for the video.",
        examples=[
            "A man is crying and he says \"I shouldn't have done it. I regret everything\""
        ],
    )
    image: File = Field(
        description="Starting frame image to animate. Supported formats: JPEG, PNG, WebP. Max 30 MB.",
    )
    end_image: Optional[File] = Field(
        default=None,
        description="Optional ending frame. The video will transition from the starting image to this image.",
    )
    mode: ModeEnum = Field(
        default=ModeEnum.quality,
        description="quality uses the full Seedance 2.0 model. fast is ~20% cheaper and quicker with slightly lower fidelity.",
    )
    resolution: ResolutionEnum = Field(
        default=ResolutionEnum.p720,
        description="Video resolution. 480p for faster generation, 720p for balanced quality.",
    )
    duration: DurationEnum = Field(
        default=DurationEnum.auto,
        description="Duration of the video in seconds (4-15). Use auto to let the model decide based on the prompt.",
    )
    aspect_ratio: AspectRatioEnum = Field(
        default=AspectRatioEnum.auto,
        description="Aspect ratio of the generated video. Use auto to infer from the input image.",
    )
    generate_audio: bool = Field(
        default=True,
        description="Whether to generate synchronized audio (sound effects, ambient, lip-synced speech). Cost is the same regardless.",
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducible generation. Use None for random.",
    )


class AppOutput(BaseAppOutput):
    """Output schema for Seedance 2.0 Image-to-Video."""

    video: File = Field(description="The generated video file with optional audio.")
    seed: Optional[int] = Field(default=None, description="The seed used for generation.")


class App(BaseApp):
    """Seedance 2.0 Image-to-Video application."""

    async def setup(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Seedance 2.0 I2V initialized")

    def _build_request(self, input_data: AppInput) -> dict:
        request = {
            "prompt": input_data.prompt,
            "image_url": input_data.image.uri,
            "resolution": input_data.resolution.value,
            "duration": input_data.duration.value,
            "aspect_ratio": input_data.aspect_ratio.value,
            "generate_audio": input_data.generate_audio,
        }
        if input_data.end_image:
            request["end_image_url"] = input_data.end_image.uri
        if input_data.seed is not None and input_data.seed != -1:
            request["seed"] = input_data.seed
        return request

    async def run(self, input_data: AppInput) -> AppOutput:
        try:
            if not input_data.image.exists():
                raise RuntimeError(f"Input image does not exist at path: {input_data.image.path}")
            if input_data.end_image and not input_data.end_image.exists():
                raise RuntimeError(f"End image does not exist at path: {input_data.end_image.path}")

            setup_fal_client()

            model_id = ENDPOINT_BY_MODE[input_data.mode]
            self.logger.info(
                f"Generating video ({input_data.mode.value}) with prompt: {input_data.prompt[:100]}..."
            )
            self.logger.info(
                f"Settings: {input_data.resolution.value}, {input_data.aspect_ratio.value}, {input_data.duration.value}s, endpoint={model_id}"
            )

            request_data = self._build_request(input_data)
            result = run_fal_model(model_id, request_data, self.logger)

            video_url = result["video"]["url"]
            video_path = download_video(video_url, self.logger)

            resolution_map = {
                "480p": VideoResolution.VIDEO_RES480_P,
                "720p": VideoResolution.VIDEO_RES720_P,
            }
            width, height = VIDEO_DIMENSIONS.get(
                (input_data.resolution.value, input_data.aspect_ratio.value),
                (1280, 720),
            )
            duration_value = input_data.duration.value
            seconds = AUTO_DURATION_SECONDS if duration_value == "auto" else float(duration_value)

            output_meta = OutputMeta(
                outputs=[
                    VideoMeta(
                        width=width,
                        height=height,
                        resolution=resolution_map.get(input_data.resolution.value),
                        seconds=seconds,
                        fps=24,
                        extra={
                            "generate_audio": input_data.generate_audio,
                            "mode": input_data.mode.value,
                        },
                    )
                ]
            )

            return AppOutput(
                video=File(path=video_path),
                seed=result.get("seed"),
                output_meta=output_meta,
            )

        except Exception as e:
            self.logger.error(f"Error during video generation: {e}")
            raise RuntimeError(f"Video generation failed: {str(e)}")

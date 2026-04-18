"""
Kling Video V3 Text-to-Video

Generate videos from text prompts using Kling Video V3 via fal.ai.
Supports pro and standard tiers with 3-15s duration and native audio generation.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta
from pydantic import Field
from typing import Optional, List
from enum import Enum
import logging

from .fal_helper import setup_fal_client, run_fal_model, download_video

logging.getLogger("httpx").setLevel(logging.WARNING)


class TierEnum(str, Enum):
    """Generation tier — pro for highest quality, standard for faster/cheaper."""
    pro = "pro"
    standard = "standard"


class ShotTypeEnum(str, Enum):
    """Multi-shot type options."""
    customize = "customize"
    intelligent = "intelligent"


class AspectRatioEnum(str, Enum):
    """Video aspect ratio options."""
    ratio_16_9 = "16:9"
    ratio_9_16 = "9:16"
    ratio_1_1 = "1:1"


class DurationEnum(str, Enum):
    """Video duration options in seconds."""
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


# Video dimensions lookup: (aspect_ratio) -> (width, height) at 720p
VIDEO_DIMENSIONS = {
    "16:9": (1280, 720),
    "9:16": (720, 1280),
    "1:1": (720, 720),
}

ENDPOINT_BY_TIER = {
    TierEnum.pro: "fal-ai/kling-video/v3/pro/text-to-video",
    TierEnum.standard: "fal-ai/kling-video/v3/standard/text-to-video",
}


class AppInput(BaseAppInput):
    """Input schema for Kling Video V3 Text-to-Video."""

    prompt: str = Field(
        description="The text prompt used to generate the video. Describe the scene, subject, and motion.",
        examples=[
            "A golden retriever running through a sunlit meadow with butterflies, cinematic slow motion."
        ],
    )
    tier: TierEnum = Field(
        default=TierEnum.pro,
        description="pro for highest quality generation, standard for faster results.",
    )
    duration: DurationEnum = Field(
        default=DurationEnum.s5,
        description="Duration of the video in seconds (3-15).",
    )
    aspect_ratio: AspectRatioEnum = Field(
        default=AspectRatioEnum.ratio_16_9,
        description="The aspect ratio of the generated video.",
    )
    shot_type: ShotTypeEnum = Field(
        default=ShotTypeEnum.intelligent,
        description="Multi-shot type. Use intelligent for automatic shot composition or customize for manual control.",
    )
    cfg_scale: float = Field(
        default=0.5,
        description="Classifier free guidance scale. Higher values follow the prompt more closely.",
    )
    negative_prompt: str = Field(
        default="blur, distort, and low quality",
        description="Negative prompt to guide what to avoid in the generated video.",
    )
    generate_audio: bool = Field(
        default=True,
        description="Whether to generate native synchronized audio for the video.",
    )
    multi_prompt: Optional[List[str]] = Field(
        default=None,
        description="List of prompts for multi-shot generation. Used when shot_type is customize.",
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducible generation. Use None for random.",
    )


class AppOutput(BaseAppOutput):
    """Output schema for Kling Video V3 Text-to-Video."""

    video: File = Field(description="The generated video file with optional audio.")
    seed: Optional[int] = Field(default=None, description="The seed used for generation.")


class App(BaseApp):
    """Kling Video V3 Text-to-Video application."""

    async def setup(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Kling Video V3 T2V initialized")

    def _build_request(self, input_data: AppInput) -> dict:
        request = {
            "prompt": input_data.prompt,
            "duration": input_data.duration.value,
            "aspect_ratio": input_data.aspect_ratio.value,
            "shot_type": input_data.shot_type.value,
            "cfg_scale": input_data.cfg_scale,
            "negative_prompt": input_data.negative_prompt,
            "generate_audio": input_data.generate_audio,
        }
        if input_data.multi_prompt is not None:
            request["multi_prompt"] = input_data.multi_prompt
        if input_data.seed is not None and input_data.seed != -1:
            request["seed"] = input_data.seed
        return request

    async def run(self, input_data: AppInput) -> AppOutput:
        try:
            setup_fal_client()

            model_id = ENDPOINT_BY_TIER[input_data.tier]
            self.logger.info(
                f"Generating video ({input_data.tier.value}) with prompt: {input_data.prompt[:100]}..."
            )
            self.logger.info(
                f"Settings: {input_data.aspect_ratio.value}, {input_data.duration.value}s, audio={input_data.generate_audio}, endpoint={model_id}"
            )

            request_data = self._build_request(input_data)
            result = run_fal_model(model_id, request_data, self.logger)

            video_url = result["video"]["url"]
            video_path = download_video(video_url, self.logger)

            width, height = VIDEO_DIMENSIONS.get(
                input_data.aspect_ratio.value,
                (1280, 720),
            )
            seconds = float(input_data.duration.value)

            output_meta = OutputMeta(
                outputs=[
                    VideoMeta(
                        width=width,
                        height=height,
                        seconds=seconds,
                        fps=24,
                        extra={
                            "generate_audio": input_data.generate_audio,
                            "tier": input_data.tier.value,
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

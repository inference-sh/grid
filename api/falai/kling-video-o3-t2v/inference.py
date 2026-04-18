"""
Kling O3 Text-to-Video

Generate videos from text prompts using Kling O3 Pro model via fal.ai.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta
from pydantic import Field
from typing import Optional
from enum import Enum
import logging

from .fal_helper import setup_fal_client, run_fal_model, download_video

logging.getLogger("httpx").setLevel(logging.WARNING)

MODEL_ENDPOINT = "fal-ai/kling-video/o3/pro/text-to-video"

VIDEO_DIMENSIONS = {
    "16:9": (1280, 720),
    "9:16": (720, 1280),
    "1:1": (720, 720),
}


class AspectRatioEnum(str, Enum):
    ratio_16_9 = "16:9"
    ratio_9_16 = "9:16"
    ratio_1_1 = "1:1"


class ShotTypeEnum(str, Enum):
    customize = "customize"
    intelligent = "intelligent"


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
    """Input schema for Kling O3 Text-to-Video."""

    prompt: Optional[str] = Field(
        default=None,
        description="The text prompt describing the video to generate.",
        examples=["A cinematic drone shot over a misty mountain range at sunrise"],
    )
    aspect_ratio: AspectRatioEnum = Field(
        default=AspectRatioEnum.ratio_16_9,
        description="Aspect ratio of the generated video.",
    )
    duration: DurationEnum = Field(
        default=DurationEnum.s5,
        description="Duration of the video in seconds (3-15).",
    )
    shot_type: Optional[ShotTypeEnum] = Field(
        default=None,
        description="Shot type control. 'customize' for manual prompt control, 'intelligent' for AI-driven camera work.",
    )
    generate_audio: bool = Field(
        default=False,
        description="Whether to generate audio for the video.",
    )


class AppOutput(BaseAppOutput):
    """Output schema for Kling O3 Text-to-Video."""

    video: File = Field(description="The generated video file.")


class App(BaseApp):
    """Kling O3 Text-to-Video application."""

    async def setup(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Kling O3 T2V initialized")

    def _build_request(self, input_data: AppInput) -> dict:
        request = {
            "aspect_ratio": input_data.aspect_ratio.value,
            "duration": input_data.duration.value,
            "generate_audio": input_data.generate_audio,
        }
        if input_data.prompt:
            request["prompt"] = input_data.prompt
        if input_data.shot_type:
            request["shot_type"] = input_data.shot_type.value
        return request

    async def run(self, input_data: AppInput) -> AppOutput:
        try:
            setup_fal_client()

            prompt_preview = input_data.prompt[:100] if input_data.prompt else "(no prompt)"
            self.logger.info(f"Generating video with prompt: {prompt_preview}...")
            self.logger.info(
                f"Settings: {input_data.aspect_ratio.value}, {input_data.duration.value}s, audio={input_data.generate_audio}"
            )

            request_data = self._build_request(input_data)
            result = run_fal_model(MODEL_ENDPOINT, request_data, self.logger)

            video_url = result["video"]["url"]
            video_path = download_video(video_url, self.logger)

            width, height = VIDEO_DIMENSIONS.get(input_data.aspect_ratio.value, (1280, 720))
            seconds = float(input_data.duration.value)

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
            self.logger.error(f"Error during video generation: {e}")
            raise RuntimeError(f"Video generation failed: {str(e)}")

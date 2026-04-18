"""
Kling O3 Image-to-Video

Generate videos from a starting image using Kling O3 model via fal.ai.
Supports pro and standard tiers, with optional end image.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta
from pydantic import Field
from typing import Optional
from enum import Enum
import logging

from .fal_helper import setup_fal_client, run_fal_model, download_video

logging.getLogger("httpx").setLevel(logging.WARNING)


class TierEnum(str, Enum):
    pro = "pro"
    standard = "standard"


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
    """Input schema for Kling O3 Image-to-Video."""

    image: File = Field(
        description="Starting frame image to animate. Supported formats: JPEG, PNG, WebP.",
    )
    prompt: Optional[str] = Field(
        default=None,
        description="The text prompt describing the desired motion and action for the video.",
        examples=["The camera slowly zooms in as the person smiles"],
    )
    end_image: Optional[File] = Field(
        default=None,
        description="Optional ending frame. The video will transition from the starting image to this image.",
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
    tier: TierEnum = Field(
        default=TierEnum.pro,
        description="Model tier. Pro offers higher quality, standard is faster and cheaper.",
    )


class AppOutput(BaseAppOutput):
    """Output schema for Kling O3 Image-to-Video."""

    video: File = Field(description="The generated video file.")


class App(BaseApp):
    """Kling O3 Image-to-Video application."""

    async def setup(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Kling O3 I2V initialized")

    def _build_request(self, input_data: AppInput) -> dict:
        request = {
            "image_url": input_data.image.uri,
            "duration": input_data.duration.value,
            "generate_audio": input_data.generate_audio,
        }
        if input_data.prompt:
            request["prompt"] = input_data.prompt
        if input_data.end_image:
            request["end_image_url"] = input_data.end_image.uri
        if input_data.shot_type:
            request["shot_type"] = input_data.shot_type.value
        return request

    async def run(self, input_data: AppInput) -> AppOutput:
        try:
            if not input_data.image.exists():
                raise RuntimeError(f"Input image does not exist at path: {input_data.image.path}")
            if input_data.end_image and not input_data.end_image.exists():
                raise RuntimeError(f"End image does not exist at path: {input_data.end_image.path}")

            setup_fal_client()

            model_id = f"fal-ai/kling-video/o3/{input_data.tier.value}/image-to-video"
            prompt_preview = input_data.prompt[:100] if input_data.prompt else "(no prompt)"
            self.logger.info(f"Generating video ({input_data.tier.value}) with prompt: {prompt_preview}...")
            self.logger.info(
                f"Settings: {input_data.duration.value}s, audio={input_data.generate_audio}, endpoint={model_id}"
            )

            request_data = self._build_request(input_data)
            result = run_fal_model(model_id, request_data, self.logger)

            video_url = result["video"]["url"]
            video_path = download_video(video_url, self.logger)

            # I2V dimensions depend on input image; default to 16:9
            width, height = 1280, 720
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

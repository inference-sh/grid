"""
Kling Video V3 Image-to-Video

Generate videos from a starting image (with optional end frame) using
Kling Video V3 via fal.ai. Supports pro and standard tiers with 3-15s
duration and native audio generation.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta
from pydantic import Field
from typing import Optional
from enum import Enum
import logging

from .fal_helper import setup_fal_client, run_fal_model, download_video

logging.getLogger("httpx").setLevel(logging.WARNING)


class TierEnum(str, Enum):
    """Generation tier — pro for highest quality, standard for faster/cheaper."""
    pro = "pro"
    standard = "standard"


class ShotTypeEnum(str, Enum):
    """Shot type for camera control."""
    customize = "customize"
    intelligent = "intelligent"


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


ENDPOINT_BY_TIER = {
    TierEnum.pro: "fal-ai/kling-video/v3/pro/image-to-video",
    TierEnum.standard: "fal-ai/kling-video/v3/standard/image-to-video",
}


class AppInput(BaseAppInput):
    """Input schema for Kling Video V3 Image-to-Video."""

    image: File = Field(
        description="Starting frame image to animate.",
    )
    prompt: Optional[str] = Field(
        default=None,
        description="Text prompt for motion guidance.",
        examples=["A woman slowly turns her head and smiles"],
    )
    end_image: Optional[File] = Field(
        default=None,
        description="Optional ending frame. The video will transition from the starting image to this image.",
    )
    tier: TierEnum = Field(
        default=TierEnum.pro,
        description="Generation tier. Pro for highest quality, standard for faster generation.",
    )
    shot_type: ShotTypeEnum = Field(
        default=ShotTypeEnum.customize,
        description="Shot type for camera control. Use intelligent to let the model decide.",
    )
    duration: DurationEnum = Field(
        default=DurationEnum.s5,
        description="Duration of the video in seconds (3-15).",
    )
    cfg_scale: float = Field(
        default=0.5,
        description="Classifier-free guidance scale.",
    )
    negative_prompt: str = Field(
        default="blur, distort, and low quality",
        description="Negative prompt to avoid unwanted artifacts.",
    )
    generate_audio: bool = Field(
        default=True,
        description="Whether to generate synchronized audio.",
    )


class AppOutput(BaseAppOutput):
    """Output schema for Kling Video V3 Image-to-Video."""

    video: File = Field(description="The generated video file.")
    seed: Optional[int] = Field(default=None, description="The seed used for generation.")


class App(BaseApp):
    """Kling Video V3 Image-to-Video application."""

    async def setup(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Kling Video V3 I2V initialized")

    def _build_request(self, input_data: AppInput) -> dict:
        request = {
            "start_image_url": input_data.image.uri,
            "shot_type": input_data.shot_type.value,
            "duration": input_data.duration.value,
            "cfg_scale": input_data.cfg_scale,
            "negative_prompt": input_data.negative_prompt,
            "generate_audio": input_data.generate_audio,
        }
        if input_data.prompt:
            request["prompt"] = input_data.prompt
        if input_data.end_image:
            request["end_image_url"] = input_data.end_image.uri
        return request

    async def run(self, input_data: AppInput) -> AppOutput:
        try:
            if not input_data.image.exists():
                raise RuntimeError(f"Input image does not exist at path: {input_data.image.path}")
            if input_data.end_image and not input_data.end_image.exists():
                raise RuntimeError(f"End image does not exist at path: {input_data.end_image.path}")

            setup_fal_client()

            model_id = ENDPOINT_BY_TIER[input_data.tier]
            prompt_preview = input_data.prompt[:100] if input_data.prompt else "(no prompt)"
            self.logger.info(
                f"Generating video ({input_data.tier.value}) with prompt: {prompt_preview}"
            )
            self.logger.info(
                f"Settings: {input_data.duration.value}s, shot_type={input_data.shot_type.value}, endpoint={model_id}"
            )

            request_data = self._build_request(input_data)
            result = run_fal_model(model_id, request_data, self.logger)

            video_url = result["video"]["url"]
            video_path = download_video(video_url, self.logger)

            seconds = float(input_data.duration.value)

            output_meta = OutputMeta(
                outputs=[
                    VideoMeta(
                        width=1280,
                        height=720,
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

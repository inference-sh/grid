"""
Kling AI Avatar v2

Generate AI avatar videos from a face image and driving audio using
Kling's avatar model. Supports pro and standard quality tiers.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta
from pydantic import Field
from typing import Optional
from enum import Enum
import logging

from .fal_helper import setup_fal_client, run_fal_model, download_video

logging.getLogger("httpx").setLevel(logging.WARNING)


class TierEnum(str, Enum):
    """Quality tier — pro uses the full model, standard is faster and cheaper."""
    pro = "pro"
    standard = "standard"


ENDPOINT_BY_TIER = {
    TierEnum.pro: "fal-ai/kling-video/ai-avatar/v2/pro",
    TierEnum.standard: "fal-ai/kling-video/ai-avatar/v2/standard",
}


class AppInput(BaseAppInput):
    """Input schema for Kling AI Avatar v2."""

    image: File = Field(
        description="Avatar face image. The model will animate this face to match the audio.",
    )
    audio: File = Field(
        description="Driving audio. The avatar will lip-sync and move to this audio.",
    )
    prompt: str = Field(
        default=".",
        description="Generation prompt to guide the avatar's appearance and behavior.",
    )
    tier: TierEnum = Field(
        default=TierEnum.pro,
        description="Quality tier. Pro gives higher fidelity, standard is faster and cheaper.",
    )


class AppOutput(BaseAppOutput):
    """Output schema for Kling AI Avatar v2."""

    video: File = Field(description="The generated avatar video.")
    duration: Optional[float] = Field(default=None, description="Duration of the generated video in seconds.")


class App(BaseApp):
    """Kling AI Avatar v2 application."""

    async def setup(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Kling AI Avatar v2 initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        try:
            setup_fal_client()

            model_id = ENDPOINT_BY_TIER[input_data.tier]
            self.logger.info(
                f"Generating avatar video ({input_data.tier.value}) with prompt: {input_data.prompt[:100]}..."
            )

            request_data = {
                "image_url": input_data.image.uri,
                "audio_url": input_data.audio.uri,
                "prompt": input_data.prompt,
            }

            result = run_fal_model(model_id, request_data, self.logger)

            video_url = result["video"]["url"]
            video_path = download_video(video_url, self.logger)

            duration = result.get("duration", None)

            output_meta = OutputMeta(
                outputs=[
                    VideoMeta(
                        width=1280,
                        height=720,
                        seconds=duration,
                        fps=24,
                        extra={
                            "tier": input_data.tier.value,
                        },
                    )
                ]
            )

            return AppOutput(
                video=File(path=video_path),
                duration=duration,
                output_meta=output_meta,
            )

        except Exception as e:
            self.logger.error(f"Error during avatar video generation: {e}")
            raise RuntimeError(f"Avatar video generation failed: {str(e)}")

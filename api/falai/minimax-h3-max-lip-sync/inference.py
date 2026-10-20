"""
MiniMax H3 Max Lip Sync via fal.ai

Animate a portrait image to speak an audio track. The output length follows the audio
(5 seconds to 15 minutes).
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta, ImageMeta, AudioMeta
from pydantic import Field
from typing import Optional
from enum import Enum
import logging

from .fal_helper import setup_fal_client, run_fal_model, download_video
from .h3max_common import dims_for, probe_video, FPS

logging.getLogger("httpx").setLevel(logging.WARNING)

ENDPOINT = "minimax/h3-max/lip-sync/image-to-video"


class ResolutionEnum(str, Enum):
    r480p = "480P"
    r768p = "768P"
    r1080p = "1080P"
    r2k = "2K"


class AppInput(BaseAppInput):
    image: File = Field(description="Image to animate. Aspect ratio between 0.4 and 2.5.")
    audio: File = Field(description="Speech audio to sync to, at least 5 seconds and up to 15 minutes. The output matches its length.")
    resolution: ResolutionEnum = Field(default=ResolutionEnum.r768p, description="Output resolution, at the supported aspect ratio nearest the image.")
    enable_transcription: bool = Field(default=False, description="Transcribe the audio to guide lip synchronization.")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducible generation.")


class AppOutput(BaseAppOutput):
    video: File = Field(description="Lip-synced video with the supplied soundtrack.")
    seed: Optional[int] = Field(default=None, description="Seed used for generation.")
    duration: Optional[float] = Field(default=None, description="Output duration in seconds.")


class App(BaseApp):

    async def setup(self, metadata):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        self.logger.info("MiniMax H3 Max Lip Sync initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        setup_fal_client()
        request = {
            "image_url": input_data.image.uri,
            "audio_url": input_data.audio.uri,
            "resolution": input_data.resolution.value,
            "enable_transcription": input_data.enable_transcription,
        }
        if input_data.seed is not None:
            request["seed"] = input_data.seed
        self.logger.info(f"Lip sync: {input_data.resolution.value}, transcription={input_data.enable_transcription}, endpoint: {ENDPOINT}")
        result = run_fal_model(ENDPOINT, request, self.logger)
        self.logger.info(f"fal timings: {result.get('timings')}, duration: {result.get('duration')}")
        video_path = download_video(result["video"]["url"], self.logger)

        info = probe_video(video_path, self.logger)
        seconds = float(result.get("duration") or info.get("seconds") or 0)
        width, height = (info.get("width"), info.get("height")) if info.get("width") else dims_for(input_data.resolution.value, "16:9")
        output_meta = OutputMeta(
            inputs=[ImageMeta(), AudioMeta(seconds=seconds)],
            outputs=[VideoMeta(
                width=int(width), height=int(height), resolution=input_data.resolution.value,
                seconds=seconds, fps=int(info.get("fps") or FPS),
                extra={"long_form": seconds > 15},
            )],
        )
        self.logger.info(f"Generated {width}x{height} video, {seconds}s")
        return AppOutput(video=File(path=video_path), seed=result.get("seed"), duration=result.get("duration"), output_meta=output_meta)

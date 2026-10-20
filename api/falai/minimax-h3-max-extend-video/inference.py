"""
MiniMax H3 Max Extend Video via fal.ai

Continue an existing video with new prompt-guided footage. Returns either the source plus
new footage as one video ("extended") or just the new clip with a 1.625s lead-in ("continuation").
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta, AudioMeta
from pydantic import Field
from typing import Optional, List, Literal
from enum import Enum
import logging

from .fal_helper import setup_fal_client, run_fal_model, download_video
from .h3max_common import dims_for, uris, probe_video, FPS

logging.getLogger("httpx").setLevel(logging.WARNING)

ENDPOINT = "minimax/h3-max/extend-video"


class ResolutionEnum(str, Enum):
    r480p = "480P"
    r768p = "768P"
    r1080p = "1080P"
    r2k = "2K"


class AspectRatioEnum(str, Enum):
    auto = "auto"
    r21_9 = "21:9"
    r16_9 = "16:9"
    r4_3 = "4:3"
    r1_1 = "1:1"
    r3_4 = "3:4"
    r9_16 = "9:16"


class AppInput(BaseAppInput):
    video: File = Field(description="Video to extend. 1.625s to 60s, up to 50 MB, aspect ratio between 0.4 and 2.5.")
    prompt: str = Field(
        description="What happens next. Describe the continuation, not the source.",
        examples=["The camera keeps pushing forward as the fog lifts and a lighthouse appears on the cliff."],
    )
    duration: float = Field(default=5, ge=1, le=15, description="Seconds of new footage to add (1-15).")
    output: Literal["extended", "continuation"] = Field(
        default="extended",
        description="extended: source plus new footage as one video. continuation: only the new clip, with a 1.625s repeat of the source ending.",
    )
    resolution: ResolutionEnum = Field(default=ResolutionEnum.r768p, description="Output resolution.")
    aspect_ratio: AspectRatioEnum = Field(default=AspectRatioEnum.auto, description="auto follows the source; other values crop to fit.")
    reference_audios: Optional[List[File]] = Field(
        default=None, max_length=3,
        description="Audio references guiding the new audio (2-15s each, 15s combined). The source soundtrack is kept.",
    )
    enable_prompt_expansion: bool = Field(default=True, description="Enhance the prompt using the source video for a consistent continuation.")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducible generation.")


class AppOutput(BaseAppOutput):
    video: File = Field(description="The extended video, or the continuation clip.")
    seed: Optional[int] = Field(default=None, description="Seed used for the continuation.")
    expanded_prompt: Optional[str] = Field(default=None, description="The prompt used after expansion.")
    duration: Optional[float] = Field(default=None, description="Seconds of returned video.")


class App(BaseApp):

    async def setup(self, metadata):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        self.logger.info("MiniMax H3 Max Extend Video initialized")

    def _build_request(self, input_data: AppInput) -> dict:
        request = {
            "video_url": input_data.video.uri,
            "prompt": input_data.prompt,
            "duration": input_data.duration,
            "output": input_data.output,
            "resolution": input_data.resolution.value,
            "aspect_ratio": input_data.aspect_ratio.value,
            "enable_prompt_expansion": input_data.enable_prompt_expansion,
        }
        if input_data.seed is not None:
            request["seed"] = input_data.seed
        if input_data.reference_audios:
            request["reference_audio_urls"] = uris(input_data.reference_audios)
        return request

    async def run(self, input_data: AppInput) -> AppOutput:
        setup_fal_client()
        self.logger.info(f"Extend: {input_data.duration}s new footage, output={input_data.output}, "
                         f"resolution={input_data.resolution.value}, endpoint={ENDPOINT}")
        result = run_fal_model(ENDPOINT, self._build_request(input_data), self.logger)
        self.logger.info(f"fal timings: {result.get('timings')}, source: {result.get('source')}, returned duration: {result.get('duration')}")
        video_path = download_video(result["video"]["url"], self.logger)

        info = probe_video(video_path, self.logger)
        width, height = (info.get("width"), info.get("height")) if info.get("width") else dims_for(input_data.resolution.value, input_data.aspect_ratio.value)
        # Billing follows the requested seconds of NEW footage, not the returned length.
        output_meta = OutputMeta(
            inputs=[VideoMeta()] + [AudioMeta() for _ in (input_data.reference_audios or [])],
            outputs=[VideoMeta(
                width=int(width), height=int(height), resolution=input_data.resolution.value,
                seconds=float(input_data.duration), fps=int(info.get("fps") or FPS),
                extra={"output": input_data.output, "returned_seconds": result.get("duration")},
            )],
        )
        return AppOutput(
            video=File(path=video_path), seed=result.get("seed"),
            expanded_prompt=result.get("expanded_prompt"), duration=result.get("duration"),
            output_meta=output_meta,
        )

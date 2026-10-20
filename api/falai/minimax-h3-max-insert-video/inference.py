"""
MiniMax H3 Max Insert Video via fal.ai

Insert a newly generated scene into an existing video: the source plays until start_time,
the generated scene runs for `duration`, then the original resumes from resume_time.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta, ImageMeta
from pydantic import Field, model_validator
from typing import Optional, List
from enum import Enum
import logging

from .fal_helper import setup_fal_client, run_fal_model, download_video
from .h3max_common import dims_for, uris, probe_video, FPS

logging.getLogger("httpx").setLevel(logging.WARNING)

ENDPOINT = "minimax/h3-max/insert-video"


class ResolutionEnum(str, Enum):
    r480p = "480p"
    r768p = "768p"


class AppInput(BaseAppInput):
    video: File = Field(description="Source video to insert a scene into.")
    start_time: float = Field(ge=1.625, le=60, description="Seconds into the source where the generated scene begins.")
    resume_time: float = Field(ge=1.625, le=60, description="Seconds into the source where the original footage resumes. Must be after start_time, with footage remaining after it.")
    duration: float = Field(default=5, ge=5, le=13, description="Length of the generated scene in seconds (5-13).")
    prompt: Optional[str] = Field(default=None, description="What happens in the inserted scene.", examples=["A red fox darts across the path and vanishes into the bushes."])
    reference_images: Optional[List[File]] = Field(default=None, max_length=9, description="Reference images guiding the inserted scene (max 9).")
    reference_videos: Optional[List[File]] = Field(default=None, max_length=3, description="Reference videos guiding the inserted scene (max 3).")
    resolution: ResolutionEnum = Field(default=ResolutionEnum.r768p, description="Output resolution.")
    enable_prompt_expansion: bool = Field(default=True, description="Enhance the prompt using the source video.")
    color_match: bool = Field(default=True, description="Match the inserted scene's color to the source footage.")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducible generation.")

    @model_validator(mode="after")
    def _check_times(self):
        if self.resume_time <= self.start_time:
            raise ValueError("resume_time must be later than start_time")
        return self


class AppOutput(BaseAppOutput):
    video: File = Field(description="The final video with the inserted scene.")
    seed: Optional[int] = Field(default=None, description="Seed used for generation.")
    expanded_prompt: Optional[str] = Field(default=None, description="The prompt used after expansion.")
    duration: Optional[float] = Field(default=None, description="Total length of the final video in seconds.")
    injected_duration: Optional[float] = Field(default=None, description="Length of the generated scene in seconds.")


class App(BaseApp):

    async def setup(self, metadata):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        self.logger.info("MiniMax H3 Max Insert Video initialized")

    def _build_request(self, input_data: AppInput) -> dict:
        request = {
            "video_url": input_data.video.uri,
            "start_time": input_data.start_time,
            "resume_time": input_data.resume_time,
            "duration": input_data.duration,
            "resolution": input_data.resolution.value,
            "enable_prompt_expansion": input_data.enable_prompt_expansion,
            "color_match": input_data.color_match,
        }
        if input_data.prompt:
            request["prompt"] = input_data.prompt
        if input_data.reference_images:
            request["reference_image_urls"] = uris(input_data.reference_images)
        if input_data.reference_videos:
            request["reference_video_urls"] = uris(input_data.reference_videos)
        if input_data.seed is not None:
            request["seed"] = input_data.seed
        return request

    async def run(self, input_data: AppInput) -> AppOutput:
        setup_fal_client()
        self.logger.info(f"Insert: start={input_data.start_time}s resume={input_data.resume_time}s scene={input_data.duration}s, "
                         f"{input_data.resolution.value}, endpoint: {ENDPOINT}")
        result = run_fal_model(ENDPOINT, self._build_request(input_data), self.logger)
        self.logger.info(f"fal timings: {result.get('timings')}, total {result.get('duration')}s, injected {result.get('injected_duration')}s")
        video_path = download_video(result["video"]["url"], self.logger)

        info = probe_video(video_path, self.logger)
        width = result.get("width") or info.get("width")
        height = result.get("height") or info.get("height")
        if not width:
            width, height = dims_for(input_data.resolution.value, "16:9")
        # Billing follows the generated scene length, not the full output.
        injected = float(result.get("injected_duration") or input_data.duration)
        output_meta = OutputMeta(
            inputs=[VideoMeta()] + [ImageMeta() for _ in (input_data.reference_images or [])] + [VideoMeta() for _ in (input_data.reference_videos or [])],
            outputs=[VideoMeta(width=int(width), height=int(height), resolution=input_data.resolution.value,
                               seconds=injected, fps=int(info.get("fps") or FPS),
                               extra={"total_seconds": result.get("duration")})],
        )
        return AppOutput(
            video=File(path=video_path), seed=result.get("seed"), expanded_prompt=result.get("expanded_prompt"),
            duration=result.get("duration"), injected_duration=result.get("injected_duration"), output_meta=output_meta,
        )

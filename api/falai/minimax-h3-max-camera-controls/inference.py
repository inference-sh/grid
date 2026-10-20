"""
MiniMax H3 Max Camera Controls via fal.ai

Turn a single image into a video with keyframed camera orbit (azimuth), elevation and distance.
The scene stays frozen; only the camera moves.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta, ImageMeta
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum
import logging

from .fal_helper import setup_fal_client, run_fal_model, download_video
from .h3max_common import dims_for, probe_video, FPS

logging.getLogger("httpx").setLevel(logging.WARNING)

ENDPOINT = "minimax/h3-max/camera-controls"
DEFAULT_PROMPT = ("The same elements in the reference are rigid. Preserve every element exactly. "
                  "The entire scene is frozen. Only the camera moves. no scene motion only camera motion")


class ResolutionEnum(str, Enum):
    r480p = "480P"
    r768p = "768P"
    r1080p = "1080P"


class PromptExpansionEnum(str, Enum):
    disabled = "disabled"
    balanced = "balanced"
    quality = "quality"


class CameraKeyframe(BaseModel):
    time: float = Field(ge=0, le=1, description="Normalized video time, 0 at the start to 1 at the end.")
    azimuth: float = Field(description="Horizontal camera angle around the subject, in degrees. Signed full turns are preserved.")
    elevation: float = Field(ge=-90, le=90, description="Vertical camera angle around the subject, in degrees.")
    distance: float = Field(ge=0, description="Camera distance from the subject in normalized scene units.")


class AppInput(BaseAppInput):
    image: File = Field(description="Image to use as the first frame.")
    camera_trajectory: List[CameraKeyframe] = Field(
        min_length=1, max_length=12,
        description="Ordered camera keyframes (max 12). The first pose is held before its time and the last pose for the remainder.",
        examples=[[{"time": 0, "azimuth": 0, "elevation": 10, "distance": 1.0}, {"time": 1, "azimuth": 90, "elevation": 10, "distance": 1.0}]],
    )
    prompt: Optional[str] = Field(
        default=None,
        description="Optional prompt. Defaults to preserving the frozen scene while only the camera moves.",
    )
    duration: float = Field(default=5, ge=1, le=15, description="Video duration in seconds (1-15).")
    resolution: ResolutionEnum = Field(default=ResolutionEnum.r480p, description="480P native (default), 768P native, 1080P latent refinement.")
    prompt_expansion_mode: PromptExpansionEnum = Field(default=PromptExpansionEnum.balanced, description="Prompt rewriting effort.")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducible generation.")


class AppOutput(BaseAppOutput):
    video: File = Field(description="The generated video.")
    expanded_prompt: Optional[str] = Field(default=None, description="The prompt after expansion.")


class App(BaseApp):

    async def setup(self, metadata):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        self.logger.info("MiniMax H3 Max Camera Controls initialized")

    def _build_request(self, input_data: AppInput) -> dict:
        request = {
            "image_url": input_data.image.uri,
            "camera_trajectory": [k.model_dump() for k in sorted(input_data.camera_trajectory, key=lambda k: k.time)],
            "prompt": input_data.prompt or DEFAULT_PROMPT,
            "duration": input_data.duration,
            "resolution": input_data.resolution.value,
            "prompt_expansion_mode": input_data.prompt_expansion_mode.value,
        }
        if input_data.seed is not None:
            request["seed"] = input_data.seed
        return request

    async def run(self, input_data: AppInput) -> AppOutput:
        setup_fal_client()
        self.logger.info(f"Camera controls: {len(input_data.camera_trajectory)} keyframes, {input_data.resolution.value}, "
                         f"{input_data.duration}s, endpoint: {ENDPOINT}")
        result = run_fal_model(ENDPOINT, self._build_request(input_data), self.logger)
        self.logger.info(f"fal timings: {result.get('timings')}")
        video_path = download_video(result["video"]["url"], self.logger)

        info = probe_video(video_path, self.logger)
        width, height = (info.get("width"), info.get("height")) if info.get("width") else dims_for(input_data.resolution.value, "16:9")
        output_meta = OutputMeta(
            inputs=[ImageMeta()],
            outputs=[VideoMeta(
                width=int(width), height=int(height), resolution=input_data.resolution.value,
                seconds=float(input_data.duration), fps=int(info.get("fps") or FPS),
                extra={"keyframes": len(input_data.camera_trajectory)},
            )],
        )
        return AppOutput(video=File(path=video_path), expanded_prompt=result.get("expanded_prompt"), output_meta=output_meta)

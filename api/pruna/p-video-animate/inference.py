"""
P-Video-Animate - Animate a reference subject with motion from a source video by Pruna
"""

import logging
from enum import Enum
from typing import Optional

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta, VideoMeta
from pydantic import Field

from .pruna_helper import run_prediction, get_generation_url, download_video, file_url, probe_video


class ResolutionEnum(str, Enum):
    hd = "720p"
    full_hd = "1080p"


class FpsEnum(str, Enum):
    original = "original"
    fps_24 = "24"
    fps_48 = "48"


class AppInput(BaseAppInput):
    video: File = Field(description="Source video (.mp4): the motion and audio source.")
    image: File = Field(description="Reference image of the subject to animate.")
    instruction_prompt: Optional[str] = Field(
        default=None,
        description="Further instructions on how the subject should be animated with the source motion.",
    )
    resolution: ResolutionEnum = Field(
        default=ResolutionEnum.hd,
        description="Pixel budget: 720p (~1 MP) or 1080p (~2 MP), aspect ratio kept.",
    )
    target_fps: FpsEnum = Field(default=FpsEnum.original, description="Frame rate of the working video.")
    turbo: bool = Field(default=False, description="Faster generation for slightly lower quality.")
    save_audio: bool = Field(default=True, description="Save the video with audio.")
    ignore_audio: bool = Field(
        default=False,
        description="Ignore source audio during generation. With save_audio the source audio is still saved.",
    )
    seed: Optional[int] = Field(default=None, description="Random seed for reproducible generation.")
    disable_safety_checker: bool = Field(default=False, description="Disable the safety checker for generated videos.")


class AppOutput(BaseAppOutput):
    video: File = Field(description="Animated video.")
    seed: Optional[int] = Field(default=None, description="Seed used for generation.")


class App(BaseApp):
    async def setup(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.model = "p-video-animate"

    async def run(self, input_data: AppInput) -> AppOutput:
        request_data = {
            "video": file_url(input_data.video, logger=self.logger),
            "image": file_url(input_data.image, logger=self.logger),
            "resolution": input_data.resolution.value,
            "target_fps": input_data.target_fps.value,
            "turbo": input_data.turbo,
            "save_audio": input_data.save_audio,
            "ignore_audio": input_data.ignore_audio,
            "disable_safety_checker": input_data.disable_safety_checker,
        }
        if input_data.instruction_prompt:
            request_data["instruction_prompt"] = input_data.instruction_prompt
        if input_data.seed is not None:
            request_data["seed"] = input_data.seed

        self.logger.info("p-video-animate: %s, fps=%s, turbo=%s", input_data.resolution.value, input_data.target_fps.value, input_data.turbo)

        result = await run_prediction(model=self.model, input_data=request_data, logger=self.logger)
        self.logger.info("Prediction metadata: %s", result.get("metadata"))
        video_path = download_video(get_generation_url(result), logger=self.logger)

        info = probe_video(video_path, logger=self.logger)
        seconds = info["seconds"] or float(result.get("duration") or 0)
        if seconds <= 0:
            seconds = probe_video(input_data.video.path, logger=self.logger)["seconds"]

        output_meta = OutputMeta(
            inputs=[ImageMeta(count=1), VideoMeta(seconds=probe_video(input_data.video.path, logger=self.logger)["seconds"])],
            outputs=[
                VideoMeta(
                    width=info["width"],
                    height=info["height"],
                    resolution=input_data.resolution.value,
                    seconds=round(seconds, 3),
                    fps=int(round(info["fps"])),
                    extra={"resolution": input_data.resolution.value},
                )
            ],
        )
        self.logger.info("Animated: %sx%s, %.2fs", info["width"], info["height"], seconds)
        return AppOutput(video=File(path=video_path), seed=result.get("seed"), output_meta=output_meta)

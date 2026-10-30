"""
P-Video-2 - Text-to-video and image-to-video with audio conditioning by Pruna (LTX-2)
"""

import logging
import math
from enum import Enum
from typing import Optional

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta, AudioMeta, VideoMeta
from pydantic import Field

from .pruna_helper import run_prediction, get_generation_url, download_video, file_url, probe_video, probe_media_duration


class ResolutionEnum(str, Enum):
    hd = "720p"
    full_hd = "1080p"


class AspectRatioEnum(str, Enum):
    landscape = "16:9"
    portrait = "9:16"
    photo_landscape = "4:3"
    photo_portrait = "3:4"
    classic_landscape = "3:2"
    classic_portrait = "2:3"
    square = "1:1"


class FpsEnum(int, Enum):
    standard = 24
    high = 48


class AppInput(BaseAppInput):
    prompt: str = Field(description="Text description for video generation.")
    image: Optional[File] = Field(
        default=None,
        description="Input image for image-to-video (jpg, png, webp). When set, aspect_ratio is ignored.",
    )
    last_frame_image: Optional[File] = Field(
        default=None,
        description="Reference image for the last frame.",
    )
    audio: Optional[File] = Field(
        default=None,
        description="Audio to condition the video (flac, mp3, wav). The video takes the audio's length; duration is ignored.",
    )
    duration: Optional[int] = Field(
        default=5,
        ge=1,
        le=20,
        description="Video duration in seconds (1-20). Leave empty to let the model pick the length from the prompt.",
    )
    resolution: ResolutionEnum = Field(default=ResolutionEnum.hd, description="Video resolution: 720p or 1080p.")
    fps: FpsEnum = Field(default=FpsEnum.standard, description="Frames per second: 24 or 48.")
    aspect_ratio: AspectRatioEnum = Field(
        default=AspectRatioEnum.landscape,
        description="Aspect ratio. Ignored when an input image is provided.",
    )
    draft: bool = Field(default=False, description="Faster, lower-quality preview at a lower price.")
    save_audio: bool = Field(default=True, description="Include audio in the output video.")
    prompt_upsampling: bool = Field(default=True, description="Enhance the prompt before generation.")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducible generation.")
    disable_safety_filter: bool = Field(default=True, description="Disable the safety filter for prompts and input images.")


class AppOutput(BaseAppOutput):
    video: File = Field(description="Generated video file.")
    seed: Optional[int] = Field(default=None, description="Seed used for generation.")


class App(BaseApp):
    async def setup(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.model = "p-video-2"

    async def run(self, input_data: AppInput) -> AppOutput:
        request_data = {
            "prompt": input_data.prompt,
            "resolution": input_data.resolution.value,
            "fps": input_data.fps.value,
            "aspect_ratio": input_data.aspect_ratio.value,
            "draft": input_data.draft,
            "save_audio": input_data.save_audio,
            "prompt_upsampling": input_data.prompt_upsampling,
            "disable_safety_filter": input_data.disable_safety_filter,
        }
        if input_data.duration is not None and not input_data.audio:
            request_data["duration"] = input_data.duration
        if input_data.image:
            request_data["image"] = file_url(input_data.image, logger=self.logger)
        if input_data.last_frame_image:
            request_data["last_frame_image"] = file_url(input_data.last_frame_image, logger=self.logger)
        if input_data.audio:
            request_data["audio"] = file_url(input_data.audio, logger=self.logger)
        if input_data.seed is not None:
            request_data["seed"] = input_data.seed

        self.logger.info(
            "p-video-2: %s, draft=%s, duration=%s, image=%s, audio=%s",
            input_data.resolution.value, input_data.draft, request_data.get("duration"),
            bool(input_data.image), bool(input_data.audio),
        )

        result = await run_prediction(model=self.model, input_data=request_data, logger=self.logger)
        self.logger.info("Prediction metadata: %s", result.get("metadata"))
        video_path = download_video(get_generation_url(result), logger=self.logger)

        # Pruna bills the finished video's length in whole seconds
        info = probe_video(video_path, logger=self.logger)
        seconds = float(math.floor(info["seconds"] + 1e-6))
        if seconds <= 0:
            if input_data.audio:
                seconds = float(math.floor(probe_media_duration(input_data.audio.path, logger=self.logger)))
            elif input_data.duration:
                seconds = float(input_data.duration)
        seconds = max(seconds, 1.0)

        inputs = []
        if input_data.image:
            inputs.append(ImageMeta(count=1))
        if input_data.last_frame_image:
            inputs.append(ImageMeta(count=1))
        if input_data.audio:
            inputs.append(AudioMeta(seconds=probe_media_duration(input_data.audio.path, logger=self.logger)))

        output_meta = OutputMeta(
            inputs=inputs,
            outputs=[
                VideoMeta(
                    width=info["width"],
                    height=info["height"],
                    resolution=input_data.resolution.value,
                    seconds=seconds,
                    fps=input_data.fps.value,
                    extra={"resolution": input_data.resolution.value, "draft": input_data.draft},
                )
            ],
        )
        self.logger.info("Video: %sx%s, %.2fs probed, billed %ss", info["width"], info["height"], info["seconds"], seconds)
        return AppOutput(video=File(path=video_path), seed=result.get("seed"), output_meta=output_meta)

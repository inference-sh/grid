"""
P-Video-2-Pro-Reference - Video from a prompt with image, video and audio references by Pruna (MiniMax H3)
"""

import logging
import math
from enum import Enum
from typing import List, Optional

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta, AudioMeta, VideoMeta
from pydantic import Field, model_validator

from .pruna_helper import run_prediction, get_generation_url, download_video, file_url, probe_video, probe_media_duration


MAX_IMAGES = 9
MAX_VIDEOS = 3
MAX_AUDIOS = 3
REFERENCE_VIDEO_CAP = 15.0


class ResolutionEnum(str, Enum):
    r480 = "480p"
    r768 = "768p"
    r1080 = "1080p"
    r2k = "2k"
    r4k = "4k"


class ModeEnum(str, Enum):
    cost = "cost"
    speed = "speed"


class PromptUpsamplerEnum(str, Enum):
    off = "off"
    turbo = "turbo"
    max = "max"


class AspectRatioEnum(str, Enum):
    adaptive = "adaptive"
    landscape = "16:9"
    portrait = "9:16"
    photo_landscape = "4:3"
    photo_portrait = "3:4"
    classic_landscape = "3:2"
    classic_portrait = "2:3"
    square = "1:1"


class AppInput(BaseAppInput):
    prompt: str = Field(description="Text description for video generation. Refer to references by what they show.")
    images: List[File] = Field(
        default_factory=list,
        max_length=MAX_IMAGES,
        json_schema_extra={"x-promoted": True},
        description="Reference images (up to 9 including first and last frames; 20 MiB, 4096 px each).",
    )
    videos: List[File] = Field(
        default_factory=list,
        max_length=MAX_VIDEOS,
        json_schema_extra={"x-promoted": True},
        description="Reference videos (up to 3; 2-15s each, 15s combined, up to 1080p/60 fps).",
    )
    audios: List[File] = Field(
        default_factory=list,
        max_length=MAX_AUDIOS,
        description="Reference audio (up to 3, 15s combined). Set keep_audio to use one file as the soundtrack.",
    )
    keep_audio: bool = Field(
        default=False,
        description="Keep the single audio file as the video's audio. Needs exactly one audio file.",
    )
    first_frame_image: Optional[File] = Field(default=None, description="Optional opening image (soft keyframe, counts as a reference image).")
    last_frame_image: Optional[File] = Field(default=None, description="Optional closing image (soft keyframe, counts as a reference image).")
    duration: int = Field(default=5, ge=5, le=15, description="Video duration in seconds (5-15).")
    resolution: ResolutionEnum = Field(
        default=ResolutionEnum.r768,
        description="480p or 768p native. 1080p, 2k and 4k generate at 768p and upscale.",
    )
    mode: ModeEnum = Field(default=ModeEnum.speed, description="speed (default) or cost (cheaper, slower).")
    prompt_upsampler: PromptUpsamplerEnum = Field(
        default=PromptUpsamplerEnum.turbo,
        description="Rewrite the prompt before generation: turbo, max or off. Frame images force turbo.",
    )
    aspect_ratio: AspectRatioEnum = Field(
        default=AspectRatioEnum.landscape,
        description="Aspect ratio. adaptive follows the references.",
    )
    seed: Optional[int] = Field(default=None, description="Random seed for reproducible generation.")
    enable_safety_filter: bool = Field(
        default=False,
        description="Check inputs against the content policy. Always-on categories are blocked either way.",
    )

    @model_validator(mode="after")
    def check_references(self):
        frames = int(self.first_frame_image is not None) + int(self.last_frame_image is not None)
        if not self.images and not self.videos and not frames:
            raise ValueError("Provide at least one image or video reference")
        if len(self.images) + frames > MAX_IMAGES:
            raise ValueError(f"At most {MAX_IMAGES} images including first and last frames")
        if self.keep_audio and len(self.audios) != 1:
            raise ValueError("keep_audio needs exactly one audio file")
        return self


class AppOutput(BaseAppOutput):
    video: File = Field(description="Generated video with audio.")
    seed: Optional[int] = Field(default=None, description="Seed used for generation.")


class App(BaseApp):
    async def setup(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.model = "p-video-2-pro-reference"

    async def run(self, input_data: AppInput) -> AppOutput:
        request_data = {
            "prompt": input_data.prompt,
            "duration": input_data.duration,
            "resolution": input_data.resolution.value,
            "mode": input_data.mode.value,
            "prompt_upsampler": input_data.prompt_upsampler.value,
            "aspect_ratio": input_data.aspect_ratio.value,
            "enable_safety_filter": input_data.enable_safety_filter,
        }
        if input_data.images:
            request_data["image"] = [file_url(f, logger=self.logger) for f in input_data.images]
        if input_data.videos:
            request_data["video"] = [file_url(f, logger=self.logger) for f in input_data.videos]
        if input_data.audios:
            request_data["audio"] = [file_url(f, logger=self.logger) for f in input_data.audios]
            request_data["keep_audio"] = input_data.keep_audio
        if input_data.first_frame_image:
            request_data["first_frame_image"] = file_url(input_data.first_frame_image, logger=self.logger)
        if input_data.last_frame_image:
            request_data["last_frame_image"] = file_url(input_data.last_frame_image, logger=self.logger)
        if input_data.seed is not None:
            request_data["seed"] = input_data.seed

        self.logger.info(
            "p-video-2-pro-reference: %s, mode=%s, %ss, images=%d, videos=%d, audios=%d, frames=%d",
            input_data.resolution.value, input_data.mode.value, input_data.duration,
            len(input_data.images), len(input_data.videos), len(input_data.audios),
            int(input_data.first_frame_image is not None) + int(input_data.last_frame_image is not None),
        )

        result = await run_prediction(model=self.model, input_data=request_data, logger=self.logger)
        self.logger.info("Prediction metadata: %s", result.get("metadata"))
        video_path = download_video(get_generation_url(result), logger=self.logger)

        # Output bills whole seconds: int(num_frames / 24)
        info = probe_video(video_path, logger=self.logger)
        seconds = float(math.floor(info["seconds"] + 1e-6)) if info["seconds"] > 0 else float(input_data.duration)

        # Each reference clip bills up to output length + 0.5s; reference video is capped at 15s combined
        clip_cap = seconds + 0.5
        inputs = []
        image_count = len(input_data.images)
        image_count += int(input_data.first_frame_image is not None) + int(input_data.last_frame_image is not None)
        if image_count:
            inputs.append(ImageMeta(count=image_count))
        video_left = REFERENCE_VIDEO_CAP
        for v in input_data.videos:
            billed = min(probe_video(v.path, logger=self.logger)["seconds"], clip_cap, video_left)
            video_left -= billed
            inputs.append(VideoMeta(seconds=round(billed, 3)))
        for a in input_data.audios:
            billed = min(probe_media_duration(a.path, logger=self.logger), clip_cap)
            inputs.append(AudioMeta(seconds=round(billed, 3)))

        output_meta = OutputMeta(
            inputs=inputs,
            outputs=[
                VideoMeta(
                    width=info["width"],
                    height=info["height"],
                    resolution=input_data.resolution.value,
                    seconds=seconds,
                    fps=24,
                    extra={"resolution": input_data.resolution.value, "mode": input_data.mode.value},
                )
            ],
        )
        self.logger.info("Video: %sx%s, %.2fs probed, billed %ss; inputs %s", info["width"], info["height"], info["seconds"], seconds,
                         [(m.type, getattr(m, "seconds", None), getattr(m, "count", None)) for m in inputs])
        return AppOutput(video=File(path=video_path), seed=result.get("seed"), output_meta=output_meta)

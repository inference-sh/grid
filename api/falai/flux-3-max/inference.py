"""
FLUX 3 Max via fal.ai

fal's post-trained FLUX 3 Video, built with Black Forest Labs: stronger prompt adherence,
better visual quality, low latency. Unified app: text-to-video, image-to-video (with
optional end frame), and keyframes-to-video (up to 10 images pinned to frame positions).
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta, ImageMeta
from pydantic import BaseModel, Field, model_validator
from typing import Optional, List, Literal
from enum import Enum
import logging

from .fal_helper import setup_fal_client, run_fal_model, download_video

logging.getLogger("httpx").setLevel(logging.WARNING)

FPS = 24


class ResolutionEnum(str, Enum):
    r720p = "720p"
    r1080p = "1080p"


class AspectRatioEnum(str, Enum):
    auto = "auto"
    r21_9 = "21:9"
    r2_1 = "2:1"
    r16_9 = "16:9"
    r4_3 = "4:3"
    r1_1 = "1:1"
    r3_4 = "3:4"
    r9_16 = "9:16"


# (resolution, aspect_ratio) -> (width, height); used when fal does not report dimensions.
DIMENSION_MAP = {
    ("720p", "16:9"): (1280, 720), ("720p", "9:16"): (720, 1280),
    ("720p", "4:3"): (960, 720), ("720p", "3:4"): (720, 960),
    ("720p", "1:1"): (720, 720), ("720p", "21:9"): (1680, 720),
    ("720p", "2:1"): (1440, 720),
    ("1080p", "16:9"): (1920, 1080), ("1080p", "9:16"): (1080, 1920),
    ("1080p", "4:3"): (1440, 1080), ("1080p", "3:4"): (1080, 1440),
    ("1080p", "1:1"): (1080, 1080), ("1080p", "21:9"): (2520, 1080),
    ("1080p", "2:1"): (2160, 1080),
}

ENDPOINTS = {
    "text-to-video": "blackforestlabs/flux-3-max/text-to-video",
    "image-to-video": "blackforestlabs/flux-3-max/image-to-video",
    "keyframes-to-video": "blackforestlabs/flux-3-max/keyframes-to-video",
}


class Keyframe(BaseModel):
    image: File = Field(description="Keyframe image (PNG, JPEG, or WebP).")
    frame_index: int = Field(
        ge=0,
        description="Frame position in the 24 fps output. Unique per keyframe and at most duration * 24.",
    )


class AppInput(BaseAppInput):
    """FLUX 3 Max video generation.

    Mode is determined by the inputs:
    - Text-to-video: prompt only
    - Image-to-video: prompt + image (optional end_image as the last frame)
    - Keyframes-to-video: prompt + keyframes (cannot combine with image/end_image)
    """

    prompt: str = Field(
        description="Text prompt describing the video. For image modes, describe how the video unfolds from the image(s).",
        examples=["A cinematic tracking shot through a misty forest at dawn, golden light filtering through ancient trees."],
    )
    image: Optional[File] = Field(
        default=None,
        description="First frame image for image-to-video (PNG, JPEG, or WebP).",
    )
    end_image: Optional[File] = Field(
        default=None,
        description="Optional last frame image (PNG, JPEG, or WebP). Requires image.",
    )
    keyframes: Optional[List[Keyframe]] = Field(
        default=None,
        max_length=10,
        description="Keyframe images pinned to frame positions (max 10, unique frame_index values). Cannot combine with image or end_image.",
    )
    duration: int = Field(
        default=5,
        ge=5,
        le=20,
        description="Video duration in seconds (5-20).",
    )
    resolution: ResolutionEnum = Field(
        default=ResolutionEnum.r720p,
        description="Output resolution. 720p is cheaper and faster; 1080p for best quality.",
    )
    aspect_ratio: AspectRatioEnum = Field(
        default=AspectRatioEnum.auto,
        description="Aspect ratio. auto uses 16:9 for text-to-video and follows the first image otherwise.",
    )
    generate_audio: bool = Field(
        default=True,
        description="Generate a native soundtrack for the video.",
    )
    reasoning_level: Literal["low", "high"] = Field(
        default="high",
        description="How much effort to spend reasoning about the prompt. high is slower but follows the prompt more closely.",
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducible generation.",
    )

    @model_validator(mode="after")
    def _check_mode(self):
        if self.keyframes is not None and len(self.keyframes) == 0:
            raise ValueError("keyframes must contain at least one keyframe when provided")
        if self.keyframes and (self.image or self.end_image):
            raise ValueError("keyframes cannot be combined with image or end_image")
        if self.end_image and not self.image:
            raise ValueError("end_image requires image to be set as the first frame")
        if self.keyframes:
            indices = [k.frame_index for k in self.keyframes]
            if len(set(indices)) != len(indices):
                raise ValueError("keyframe frame_index values must be unique")
            max_index = self.duration * FPS
            if max(indices) > max_index:
                raise ValueError(f"keyframe frame_index must be at most duration * {FPS} ({max_index})")
        return self


class AppOutput(BaseAppOutput):
    video: File = Field(description="The generated video file.")
    seed: Optional[int] = Field(default=None, description="The seed used for generation.")


class App(BaseApp):

    async def setup(self, metadata):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        self.logger.info("FLUX 3 Max initialized")

    def _determine_mode(self, input_data: AppInput) -> str:
        if input_data.keyframes:
            return "keyframes-to-video"
        if input_data.image:
            return "image-to-video"
        return "text-to-video"

    def _build_request(self, input_data: AppInput, mode: str) -> dict:
        request = {
            "prompt": input_data.prompt,
            "duration": input_data.duration,
            "resolution": input_data.resolution.value,
            "aspect_ratio": input_data.aspect_ratio.value,
            "generate_audio": input_data.generate_audio,
            "reasoning_level": input_data.reasoning_level,
        }
        if input_data.seed is not None:
            request["seed"] = input_data.seed

        if mode == "image-to-video":
            request["image_url"] = input_data.image.uri
            if input_data.end_image:
                request["end_image_url"] = input_data.end_image.uri
        elif mode == "keyframes-to-video":
            request["keyframes"] = [
                {"image_url": k.image.uri, "frame_index": k.frame_index}
                for k in sorted(input_data.keyframes, key=lambda k: k.frame_index)
            ]
        return request

    async def run(self, input_data: AppInput) -> AppOutput:
        setup_fal_client()

        mode = self._determine_mode(input_data)
        endpoint = ENDPOINTS[mode]
        self.logger.info(
            f"Mode: {mode}, resolution: {input_data.resolution.value}, "
            f"duration: {input_data.duration}s, aspect_ratio: {input_data.aspect_ratio.value}, "
            f"audio: {input_data.generate_audio}, reasoning: {input_data.reasoning_level}, endpoint: {endpoint}"
        )

        request_data = self._build_request(input_data, mode)
        result = run_fal_model(endpoint, request_data, self.logger)
        self.logger.info(f"fal timings: {result.get('timings')}")

        video_info = result["video"]
        video_path = download_video(video_info["url"], self.logger)

        ratio = input_data.aspect_ratio.value
        if ratio == "auto":
            ratio = "16:9"
        fallback_w, fallback_h = DIMENSION_MAP.get((input_data.resolution.value, ratio), (1280, 720))
        width = video_info.get("width") or fallback_w
        height = video_info.get("height") or fallback_h
        # Bill on the requested duration: fal prices per requested second, and the file can run a frame or two longer.
        seconds = float(input_data.duration)
        fps = int(video_info.get("fps") or FPS)
        self.logger.info(f"fal reported {video_info.get('width')}x{video_info.get('height')}, {video_info.get('duration')}s")

        input_metas = []
        if input_data.image:
            input_metas.append(ImageMeta())
        if input_data.end_image:
            input_metas.append(ImageMeta())
        if input_data.keyframes:
            input_metas.extend(ImageMeta() for _ in input_data.keyframes)

        output_meta = OutputMeta(
            inputs=input_metas,
            outputs=[VideoMeta(
                width=width,
                height=height,
                resolution=input_data.resolution.value,
                seconds=seconds,
                fps=fps,
                extra={
                    "mode": mode,
                    "generate_audio": input_data.generate_audio,
                    "reasoning_level": input_data.reasoning_level,
                },
            )],
        )

        self.logger.info(f"Generated {width}x{height} video, {seconds}s @ {fps}fps")
        return AppOutput(
            video=File(path=video_path),
            seed=result.get("seed"),
            output_meta=output_meta,
        )

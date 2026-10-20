"""
MiniMax H3 Max via fal.ai

fal's post-trained H3 — stronger prompt adherence, better aesthetics, ~3s for a 5s clip at 768P.
Unified app: text-to-video, image-to-video (first and/or last frame), and reference-based generation.
480P/768P native, 1080P latent refinement. Optional target audio pins the soundtrack (t2v and i2v).
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta, ImageMeta, AudioMeta
from pydantic import Field, model_validator
from typing import Optional, List
from enum import Enum
import logging

from .fal_helper import setup_fal_client, run_fal_model, download_video

logging.getLogger("httpx").setLevel(logging.WARNING)


class ResolutionEnum(str, Enum):
    r480p = "480P"
    r768p = "768P"
    r1080p = "1080P"


class AspectRatioEnum(str, Enum):
    adaptive = "adaptive"
    r21_9 = "21:9"
    r16_9 = "16:9"
    r4_3 = "4:3"
    r1_1 = "1:1"
    r3_4 = "3:4"
    r9_16 = "9:16"


class PromptExpansionEnum(str, Enum):
    disabled = "disabled"
    balanced = "balanced"
    quality = "quality"


DIMENSION_MAP = {
    ("768P", "16:9"): (1365, 768), ("768P", "9:16"): (768, 1365),
    ("768P", "4:3"): (1024, 768), ("768P", "3:4"): (768, 1024),
    ("768P", "1:1"): (768, 768), ("768P", "21:9"): (1792, 768),
    ("480P", "16:9"): (854, 480), ("480P", "9:16"): (480, 854),
    ("480P", "4:3"): (640, 480), ("480P", "3:4"): (480, 640),
    ("480P", "1:1"): (480, 480), ("480P", "21:9"): (1120, 480),
    ("1080P", "16:9"): (1920, 1080), ("1080P", "9:16"): (1080, 1920),
    ("1080P", "4:3"): (1440, 1080), ("1080P", "3:4"): (1080, 1440),
    ("1080P", "1:1"): (1080, 1080), ("1080P", "21:9"): (2520, 1080),
}

ENDPOINTS = {
    "text-to-video": "minimax/h3-max/text-to-video",
    "image-to-video": "minimax/h3-max/image-to-video",
    "reference-to-video": "minimax/h3-max/reference-to-video",
}


class AppInput(BaseAppInput):
    """MiniMax H3 Max — fal's post-trained H3.

    Modes determined by inputs:
    - Text-to-video: prompt only (aspect_ratio required, no adaptive)
    - Image-to-video: prompt + image and/or end_image (first frame, last frame, or both)
    - Reference: prompt + reference images/videos/audio (aspect_ratio defaults to adaptive)
    """

    prompt: str = Field(
        description="Video prompt. Max 50000 chars. Use timeline markers like [0s-3s] for scene pacing.",
        examples=["A cinematic tracking shot through a misty forest at dawn, golden light filtering through ancient trees."],
    )
    image: Optional[File] = Field(
        default=None,
        description="First frame image for image-to-video. Formats: JPG, PNG, WEBP, GIF, AVIF.",
    )
    end_image: Optional[File] = Field(
        default=None,
        description="Last frame image. Can be given alone for end-only generation; the canvas then follows this image.",
    )
    target_audio: Optional[File] = Field(
        default=None,
        description="Audio clip (2s or longer, max 15 MB) pinned as the soundtrack. Trimmed to the video length. Text-to-video and image-to-video only; use duration of at least 2s.",
    )
    reference_images: Optional[List[File]] = Field(
        default=None,
        description="Reference images for style/subject guidance (max 9). Cannot combine with first/last frame.",
    )
    reference_videos: Optional[List[File]] = Field(
        default=None,
        description="Reference videos for motion/style (max 3). MP4, 2-15s each.",
    )
    reference_audios: Optional[List[File]] = Field(
        default=None,
        description="Reference audio clips (max 3). Requires reference image or video. 2-15s each.",
    )
    duration: float = Field(
        default=5,
        ge=1,
        le=15,
        description="Video duration in seconds (1-15). Output can run up to ~0.7s longer than requested.",
    )
    resolution: ResolutionEnum = Field(
        default=ResolutionEnum.r768p,
        description="Output resolution. 480P fastest, 768P native quality, 1080P latent refinement from 768P (slowest, priciest).",
    )
    aspect_ratio: AspectRatioEnum = Field(
        default=AspectRatioEnum.r16_9,
        description="Aspect ratio. Text-to-video requires explicit value (not adaptive). Reference mode defaults to adaptive.",
    )
    prompt_expansion_mode: PromptExpansionEnum = Field(
        default=PromptExpansionEnum.balanced,
        description="Prompt rewriting effort: disabled (none), balanced (~1s), or quality (~30s extra) for richer rewriting.",
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducible generation.",
    )

    @model_validator(mode="after")
    def _check_mode(self):
        reference = bool(self.reference_images or self.reference_videos)
        if self.reference_audios and not reference:
            raise ValueError("reference_audios requires reference_images or reference_videos")
        if self.target_audio and reference:
            raise ValueError("target_audio is not supported in reference mode; use reference_audios")
        if self.target_audio and self.duration < 2:
            raise ValueError("duration must be at least 2s when target_audio is set")
        return self


class AppOutput(BaseAppOutput):
    video: File = Field(description="The generated video file.")
    seed: Optional[int] = Field(default=None, description="The seed used for generation.")
    expanded_prompt: Optional[str] = Field(default=None, description="The prompt after expansion, as sent to the model.")


class App(BaseApp):

    async def setup(self, metadata):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        self.logger.info("MiniMax H3 Max initialized")

    def _determine_mode(self, input_data: AppInput) -> str:
        if input_data.reference_images or input_data.reference_videos:
            return "reference-to-video"
        if input_data.image or input_data.end_image:
            return "image-to-video"
        return "text-to-video"

    def _build_request(self, input_data: AppInput, mode: str) -> dict:
        request = {
            "prompt": input_data.prompt,
            "duration": input_data.duration,
            "resolution": input_data.resolution.value,
            "prompt_expansion_mode": input_data.prompt_expansion_mode.value,
        }

        if input_data.seed is not None:
            request["seed"] = input_data.seed

        if mode == "text-to-video":
            ratio = input_data.aspect_ratio.value
            if ratio == "adaptive":
                ratio = "16:9"
            request["aspect_ratio"] = ratio

        elif mode == "image-to-video":
            if input_data.image:
                request["image_url"] = input_data.image.uri
            if input_data.end_image:
                request["end_image_url"] = input_data.end_image.uri

        if mode in ("text-to-video", "image-to-video") and input_data.target_audio:
            request["target_audio_url"] = input_data.target_audio.uri

        elif mode == "reference-to-video":
            request["aspect_ratio"] = input_data.aspect_ratio.value
            if input_data.reference_images:
                request["reference_image_urls"] = [img.uri for img in input_data.reference_images]
            if input_data.reference_videos:
                request["reference_video_urls"] = [vid.uri for vid in input_data.reference_videos]
            if input_data.reference_audios:
                request["reference_audio_urls"] = [aud.uri for aud in input_data.reference_audios]

        return request

    async def run(self, input_data: AppInput) -> AppOutput:
        setup_fal_client()

        mode = self._determine_mode(input_data)
        endpoint = ENDPOINTS[mode]

        self.logger.info(f"Mode: {mode}, resolution: {input_data.resolution.value}, "
                         f"duration: {input_data.duration}s, expansion: {input_data.prompt_expansion_mode.value}, "
                         f"target_audio: {bool(input_data.target_audio)}, endpoint: {endpoint}")

        request_data = self._build_request(input_data, mode)
        result = run_fal_model(endpoint, request_data, self.logger)
        self.logger.info(f"fal timings: {result.get('timings')}")

        video_url = result["video"]["url"]
        video_path = download_video(video_url, self.logger)

        ratio = input_data.aspect_ratio.value
        if mode == "text-to-video" and ratio == "adaptive":
            ratio = "16:9"
        width, height = DIMENSION_MAP.get(
            (input_data.resolution.value, ratio),
            (1365, 768),
        )

        input_metas = []
        if input_data.image:
            input_metas.append(ImageMeta())
        if input_data.end_image:
            input_metas.append(ImageMeta())
        if input_data.target_audio:
            input_metas.append(AudioMeta())
        if input_data.reference_images:
            for _ in input_data.reference_images:
                input_metas.append(ImageMeta())
        if input_data.reference_videos:
            for _ in input_data.reference_videos:
                input_metas.append(VideoMeta())
        if input_data.reference_audios:
            for _ in input_data.reference_audios:
                input_metas.append(AudioMeta())

        output_meta = OutputMeta(
            inputs=input_metas,
            outputs=[VideoMeta(
                width=width,
                height=height,
                resolution=input_data.resolution.value,
                seconds=float(input_data.duration),
                fps=24,
                extra={
                    "mode": mode,
                    "prompt_expansion_mode": input_data.prompt_expansion_mode.value,
                },
            )],
        )

        self.logger.info(f"Generated {width}x{height} video, {input_data.duration}s")
        return AppOutput(
            video=File(path=video_path),
            seed=result.get("seed"),
            expanded_prompt=result.get("expanded_prompt"),
            output_meta=output_meta,
        )

"""
MiniMax H3 Max Turbo via fal.ai

fal's faster, cheaper post-trained H3 variant. Text-to-video and image-to-video
(first frame, last frame, or both) with optional soundtrack pinning.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta, ImageMeta, AudioMeta
from pydantic import Field, model_validator
from typing import Optional
from enum import Enum
import logging

from .fal_helper import setup_fal_client, run_fal_model, download_video
from .h3max_common import dims_for, FPS

logging.getLogger("httpx").setLevel(logging.WARNING)


class ResolutionEnum(str, Enum):
    r480p = "480P"
    r768p = "768P"
    r1080p = "1080P"


class AspectRatioEnum(str, Enum):
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


ENDPOINTS = {
    "text-to-video": "minimax/h3-max-turbo/text-to-video",
    "image-to-video": "minimax/h3-max-turbo/image-to-video",
}


class AppInput(BaseAppInput):
    """MiniMax H3 Max Turbo.

    Mode is determined by the inputs:
    - Text-to-video: prompt only
    - Image-to-video: prompt + image and/or end_image
    """

    prompt: str = Field(
        description="Video prompt. Use timeline markers like [0s-3s] for scene pacing.",
        examples=["A cinematic tracking shot through a misty forest at dawn, golden light filtering through ancient trees."],
    )
    image: Optional[File] = Field(default=None, description="First frame image. The output canvas follows this image.")
    end_image: Optional[File] = Field(
        default=None,
        description="Last frame image. Can be given alone for end-only generation; the canvas then follows this image.",
    )
    target_audio: Optional[File] = Field(
        default=None,
        description="Audio clip (2s or longer, max 15 MB) pinned as the soundtrack. Trimmed to the video length. Use duration of at least 2s.",
    )
    duration: float = Field(default=5, ge=1, le=15, description="Video duration in seconds (1-15). Output can run up to ~0.7s longer.")
    resolution: ResolutionEnum = Field(
        default=ResolutionEnum.r768p,
        description="480P fastest, 768P native quality, 1080P latent refinement from 768P.",
    )
    aspect_ratio: AspectRatioEnum = Field(
        default=AspectRatioEnum.r16_9,
        description="Aspect ratio for text-to-video. Ignored when an image is given (the canvas follows the image).",
    )
    prompt_expansion_mode: PromptExpansionEnum = Field(
        default=PromptExpansionEnum.balanced,
        description="Prompt rewriting effort: disabled (none), balanced (~1s), or quality (~30s extra).",
    )
    seed: Optional[int] = Field(default=None, description="Random seed for reproducible generation.")

    @model_validator(mode="after")
    def _check(self):
        if self.target_audio and self.duration < 2:
            raise ValueError("duration must be at least 2s when target_audio is set")
        return self


class AppOutput(BaseAppOutput):
    video: File = Field(description="The generated video file.")
    expanded_prompt: Optional[str] = Field(default=None, description="The prompt after expansion, as sent to the model.")


class App(BaseApp):

    async def setup(self, metadata):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        self.logger.info("MiniMax H3 Max Turbo initialized")

    def _determine_mode(self, input_data: AppInput) -> str:
        return "image-to-video" if (input_data.image or input_data.end_image) else "text-to-video"

    def _build_request(self, input_data: AppInput, mode: str) -> dict:
        request = {
            "prompt": input_data.prompt,
            "duration": input_data.duration,
            "resolution": input_data.resolution.value,
            "prompt_expansion_mode": input_data.prompt_expansion_mode.value,
        }
        if input_data.seed is not None:
            request["seed"] = input_data.seed
        if input_data.target_audio:
            request["target_audio_url"] = input_data.target_audio.uri
        if mode == "text-to-video":
            request["aspect_ratio"] = input_data.aspect_ratio.value
        else:
            if input_data.image:
                request["image_url"] = input_data.image.uri
            if input_data.end_image:
                request["end_image_url"] = input_data.end_image.uri
        return request

    async def run(self, input_data: AppInput) -> AppOutput:
        setup_fal_client()
        mode = self._determine_mode(input_data)
        endpoint = ENDPOINTS[mode]
        self.logger.info(f"Mode: {mode}, resolution: {input_data.resolution.value}, duration: {input_data.duration}s, "
                         f"expansion: {input_data.prompt_expansion_mode.value}, target_audio: {bool(input_data.target_audio)}, endpoint: {endpoint}")

        result = run_fal_model(endpoint, self._build_request(input_data, mode), self.logger)
        self.logger.info(f"fal timings: {result.get('timings')}")
        video_path = download_video(result["video"]["url"], self.logger)

        width, height = dims_for(input_data.resolution.value, input_data.aspect_ratio.value)
        input_metas = []
        if input_data.image:
            input_metas.append(ImageMeta())
        if input_data.end_image:
            input_metas.append(ImageMeta())
        if input_data.target_audio:
            input_metas.append(AudioMeta())

        output_meta = OutputMeta(
            inputs=input_metas,
            outputs=[VideoMeta(
                width=width, height=height, resolution=input_data.resolution.value,
                seconds=float(input_data.duration), fps=FPS,
                extra={"mode": mode, "prompt_expansion_mode": input_data.prompt_expansion_mode.value},
            )],
        )
        self.logger.info(f"Generated {input_data.resolution.value} video, {input_data.duration}s")
        return AppOutput(video=File(path=video_path), expanded_prompt=result.get("expanded_prompt"), output_meta=output_meta)

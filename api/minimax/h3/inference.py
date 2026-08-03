import logging
import os
from typing import List, Optional
from enum import Enum

import httpx
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta, ImageMeta, AudioMeta
from pydantic import Field

from .minimax_helper import create_video, poll_video, download_file


class ResolutionEnum(str, Enum):
    r768p = "768P"
    r2k = "2K"


class RatioEnum(str, Enum):
    adaptive = "adaptive"
    r21_9 = "21:9"
    r16_9 = "16:9"
    r4_3 = "4:3"
    r1_1 = "1:1"
    r3_4 = "3:4"
    r9_16 = "9:16"


class AppInput(BaseAppInput):
    """MiniMax H3 — next-gen multimodal video generation.

    Modes determined by inputs:
    - Text-to-video: prompt only (ratio required, no adaptive)
    - Image-to-video: prompt + image (first frame)
    - First-last frame: prompt + image + last_image
    - Reference: prompt + reference images/videos/audio
    """

    prompt: str = Field(
        description="Video prompt. Use six-block structure: style contract, timeline with [0s-2s] markers, camera, audio cues, spelled-out text, negative list. Max 7000 chars.",
        examples=["[0s-3s] Static wide shot. A lone marble camera on a pedestal in darkness, single moonbeam. [3s-6s] The lens flares to life, film strips unfurl from the reels. [6s-8s] Film strips fill the frame. Camera: locked off, no cuts. Audio: low hum building to orchestral swell at 6s. Do not add subtitles."],
    )
    image: Optional[File] = Field(
        default=None,
        description="First frame image for image-to-video. Formats: JPG, PNG, WEBP, HEIC. Max 30MB.",
    )
    last_image: Optional[File] = Field(
        default=None,
        description="Last frame image. Requires image to be set as first frame.",
    )
    reference_images: Optional[List[File]] = Field(
        default=None,
        description="Reference images for style/content guidance. Cannot combine with first/last frame.",
    )
    reference_video: Optional[File] = Field(
        default=None,
        description="Reference video for motion/style guidance. MP4/MOV, max 50MB, 2-15s.",
    )
    reference_audio: Optional[File] = Field(
        default=None,
        description="Reference audio for synchronized generation. WAV/MP3, max 15MB, 2-15s.",
    )
    resolution: ResolutionEnum = Field(
        default=ResolutionEnum.r2k,
        description="Output resolution. 2K is the only resolution currently accepted.",
    )
    duration: int = Field(
        default=8,
        ge=5,
        le=15,
        description="Video duration in seconds. 5-10 for text/image-to-video, 5-15 for reference-to-video.",
    )
    ratio: RatioEnum = Field(
        default=RatioEnum.r16_9,
        description="Aspect ratio. Text-to-video requires explicit value (not adaptive). Image/reference modes default to adaptive.",
    )


class AppOutput(BaseAppOutput):
    video: File = Field(description="The generated video file.")


DIMENSION_MAP = {
    ("768P", "16:9"): (1365, 768), ("768P", "9:16"): (768, 1365),
    ("768P", "4:3"): (1024, 768), ("768P", "3:4"): (768, 1024),
    ("768P", "1:1"): (1024, 1024), ("768P", "21:9"): (1792, 768),
    ("2K", "16:9"): (2560, 1440), ("2K", "9:16"): (1440, 2560),
    ("2K", "4:3"): (1920, 1440), ("2K", "3:4"): (1440, 1920),
    ("2K", "1:1"): (1920, 1920), ("2K", "21:9"): (2560, 1097),
}


def _get_api_key() -> str:
    key = os.environ.get("MINIMAX_KEY")
    if not key:
        raise RuntimeError(
            "MINIMAX_KEY is not set. Check that `belt secrets get MINIMAX_KEY "
            "--json` reports a non-empty masked_value, and re-set it if not."
        )
    return key.strip()


class App(BaseApp):

    async def setup(self, metadata):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        self.api_key = _get_api_key()
        self.client = httpx.AsyncClient(timeout=300)
        self.logger.info("MiniMax H3 initialized")

    def _build_content(self, input_data: AppInput) -> list:
        content = [{"type": "text", "text": input_data.prompt}]

        if input_data.image:
            item = {"type": "image_url", "image_url": {"url": input_data.image.uri}}
            if input_data.last_image:
                item["role"] = "first_frame"
            content.append(item)

        if input_data.last_image:
            content.append({
                "type": "image_url",
                "image_url": {"url": input_data.last_image.uri},
                "role": "last_frame",
            })

        if input_data.reference_images:
            for img in input_data.reference_images:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": img.uri},
                    "role": "reference_image",
                })

        if input_data.reference_video:
            content.append({
                "type": "video_url",
                "video_url": {"url": input_data.reference_video.uri},
                "role": "reference_video",
            })

        if input_data.reference_audio:
            content.append({
                "type": "audio_url",
                "audio_url": {"url": input_data.reference_audio.uri},
                "role": "reference_audio",
            })

        return content

    def _determine_mode(self, input_data: AppInput) -> str:
        if input_data.image and input_data.last_image:
            return "first-last-frame"
        if input_data.image:
            return "image-to-video"
        if input_data.reference_images or input_data.reference_video:
            return "reference"
        return "text-to-video"

    async def run(self, input_data: AppInput) -> AppOutput:
        content = self._build_content(input_data)
        mode = self._determine_mode(input_data)

        ratio = input_data.ratio.value
        if mode == "text-to-video" and ratio == "adaptive":
            ratio = "16:9"

        self.logger.info(f"Mode: {mode}, resolution: {input_data.resolution.value}, "
                         f"duration: {input_data.duration}s, ratio: {ratio}")

        payload = {
            "model": "MiniMax-H3",
            "content": content,
            "resolution": input_data.resolution.value,
            "duration": input_data.duration,
            "ratio": ratio,
        }

        task_id = await create_video(self.client, self.api_key, payload, self.logger)
        self.logger.info(f"Task created: {task_id}")

        result = await poll_video(self.client, self.api_key, task_id, self.logger)
        self.logger.info("Video ready, downloading...")

        video_path = await download_file(self.client, result["url"], "/tmp/output.mp4")

        usage = result.get("usage") or {}
        video_duration = float(usage.get("total_seconds") or usage.get("output_seconds") or input_data.duration)
        width, height = DIMENSION_MAP.get(
            (input_data.resolution.value, ratio),
            (1365, 768),
        )

        input_metas = []
        if input_data.image:
            input_metas.append(ImageMeta())
        if input_data.last_image:
            input_metas.append(ImageMeta())
        if input_data.reference_images:
            for _ in input_data.reference_images:
                input_metas.append(ImageMeta())
        if input_data.reference_video:
            input_metas.append(VideoMeta())
        if input_data.reference_audio:
            input_metas.append(AudioMeta())

        output_meta = OutputMeta(
            inputs=input_metas,
            outputs=[VideoMeta(
                width=width,
                height=height,
                resolution=input_data.resolution.value,
                seconds=video_duration,
                fps=24,
                extra={
                    "mode": mode,
                    "model": "MiniMax-H3",
                    "input_image_count": usage.get("input_image_count", 0),
                },
            )],
        )

        self.logger.info(f"Generated {width}x{height} video, {video_duration}s")
        return AppOutput(video=File(path=video_path), output_meta=output_meta)

    async def on_cancel(self):
        self.logger.info("Cancellation requested")
        return True

    async def unload(self):
        await self.client.aclose()

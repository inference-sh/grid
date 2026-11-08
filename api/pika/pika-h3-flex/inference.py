import logging
import os
import tempfile
from typing import List, Literal, Optional

import httpx
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta, ImageMeta, AudioMeta
from pydantic import Field, model_validator

from .pika_helper import (
    PikaError, get_api_key, upload_file, submit_job, poll_job, cancel_job,
    download_file, probe_media, charge_micro_usd, stem_seconds,
)

MODEL_PATH = "/v1/media/pika/pika-h3-flex"

Resolution = Literal["480p", "768p", "1080p", "2k", "4k"]
AspectRatio = Literal["auto", "21:9", "16:9", "3:2", "4:3", "5:4", "1:1", "4:5", "3:4", "2:3", "9:16", "9:21"]


class AppInput(BaseAppInput):
    """Pika H3 Flex video generation with synchronized audio.

    The endpoint is chosen from the inputs: a prompt alone is text-to-video;
    a first and/or last frame image is image-to-video.
    """

    prompt: str = Field(
        min_length=1,
        max_length=8000,
        description="What happens in the video. Describe shots with time ranges (0-4s, 4-8s), camera moves, subjects, lighting and the sound (dialogue, ambience, music or 'no music').",
        examples=["A red fox trots along a frozen riverbank at dawn, breath steaming. 0-3s: low tracking shot beside the fox. 3-5s: it stops and looks straight into the lens. Sound: crunching snow, distant crows, no music."],
    )
    image: Optional[File] = Field(
        default=None,
        description="First-frame image (JPEG, PNG or WebP, up to 24 MiB and 40 MP). Resized and center-cropped to the output canvas.",
    )
    last_frame_image: Optional[File] = Field(
        default=None,
        description="Last-frame image, same limits as image. Can be used alone or together with image.",
    )
    duration: int = Field(
        default=5, ge=3, le=15,
        description="Video length in whole seconds, 3 to 15. Billed per output second.",
    )
    aspect_ratio: AspectRatio = Field(
        default="auto",
        description="Output aspect ratio. auto follows the prompt, or the first keyframe when one is given.",
    )
    resolution: Resolution = Field(
        default="768p",
        description="480p and 768p are native. 1080p, 2k and 4k generate at 768p and upscale once, keeping audio and frame rate.",
    )
    rewrite: bool = Field(
        default=True,
        description="Let Pika expand the prompt before generation. Turn off to use the prompt verbatim.",
    )
    seed: int = Field(default=0, ge=0, description="Seed. 0 lets Pika choose.")
    audio_stems: bool = Field(
        default=False,
        description="Also return the soundtrack split into Dialogue, Music and Effect WAV stems. Billed per stem second on top of the video.",
    )
    stems_channel_mode: Literal["mono", "stereo"] = Field(
        default="mono",
        description="Audio stems only. mono downmixes before separation; stereo keeps both channels.",
    )


class AppOutput(BaseAppOutput):
    video: File = Field(description="Generated MP4 (H.264 + AAC, 24 fps).")
    stems: List[File] = Field(default_factory=list, description="Dialogue, Music and Effect WAV stems when audio_stems is on.")
    seed: Optional[int] = Field(default=None, description="Seed Pika used for this video.")


class App(BaseApp):

    async def setup(self, metadata):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        self.api_key = get_api_key()
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=30.0))
        self.job_id: Optional[str] = None
        self.logger.info("Pika H3 Flex initialized")

    def _determine_mode(self, input_data: AppInput) -> str:
        if input_data.image and input_data.last_frame_image:
            return "first-last-frame"
        if input_data.image:
            return "image-to-video"
        if input_data.last_frame_image:
            return "last-frame"
        return "text-to-video"

    def _build_payload(self, input_data: AppInput, image_url: Optional[str], last_url: Optional[str]) -> dict:
        payload = {
            "prompt": input_data.prompt,
            "duration": input_data.duration,
            "aspect_ratio": input_data.aspect_ratio,
            "resolution": input_data.resolution,
            "seed": input_data.seed,
            "rewrite": input_data.rewrite,
            "output_mode": "audio-stems" if input_data.audio_stems else "video",
        }
        if input_data.audio_stems:
            payload["channel_mode"] = input_data.stems_channel_mode
        if image_url:
            payload["image"] = image_url
        if last_url:
            payload["last_frame_image"] = last_url
        return payload

    async def run(self, input_data: AppInput) -> AppOutput:
        mode = self._determine_mode(input_data)
        endpoint = "text-to-video" if mode == "text-to-video" else "image-to-video"
        self.logger.info(f"Mode: {mode}, resolution: {input_data.resolution}, duration: {input_data.duration}s, "
                         f"ratio: {input_data.aspect_ratio}, stems: {input_data.audio_stems}")

        image_url = await upload_file(self.client, self.api_key, input_data.image, "image/png", self.logger) if input_data.image else None
        last_url = await upload_file(self.client, self.api_key, input_data.last_frame_image, "image/png", self.logger) if input_data.last_frame_image else None

        payload = self._build_payload(input_data, image_url, last_url)
        self.job_id = await submit_job(self.client, self.api_key, f"{MODEL_PATH}/{endpoint}", payload, self.logger)
        job = await poll_job(self.client, self.api_key, self.job_id, self.logger)
        self.job_id = None

        output = job.get("output") or {}
        video_url = (output.get("video") or {}).get("url")
        if not video_url:
            raise PikaError(f"completed job has no video url: {str(output)[:300]}")

        workdir = tempfile.mkdtemp(prefix="pika-")
        video_path = await download_file(self.client, video_url, os.path.join(workdir, "video.mp4"), self.logger)
        info = probe_media(video_path)
        self.logger.info(f"Video: {info.get('width')}x{info.get('height')} {info.get('seconds', 0):.2f}s @ {info.get('fps', 0):.0f}fps")

        stem_files: List[File] = []
        stem_metas: List[AudioMeta] = []
        if input_data.audio_stems:
            if output.get("stems_error"):
                self.logger.warning(f"Stem separation failed upstream (video delivered, stems not charged): {output['stems_error']}")
            for i, stem in enumerate(output.get("stems") or []):
                url = stem.get("url")
                if not url:
                    continue
                name = (stem.get("name") or f"stem_{i + 1}").lower().replace(" ", "_")
                path = await download_file(self.client, url, os.path.join(workdir, f"{name}.wav"), self.logger)
                secs = stem_seconds(stem) or probe_media(path).get("seconds", 0.0)
                stem_files.append(File(path=path))
                stem_metas.append(AudioMeta(seconds=secs, sample_rate=int(stem.get("sample_rate") or 0), extra={"stem": name}))
            self.logger.info(f"Stems: {[m.extra['stem'] for m in stem_metas]} total {sum(m.seconds for m in stem_metas):.2f}s")

        usage = job.get("usage") or {}
        billed_seconds = float(usage.get("output_seconds") or info.get("seconds") or input_data.duration)
        charge = charge_micro_usd(job)

        input_metas = []
        if input_data.image:
            input_metas.append(ImageMeta())
        if input_data.last_frame_image:
            input_metas.append(ImageMeta())

        output_meta = OutputMeta(
            inputs=input_metas,
            outputs=[VideoMeta(
                width=int(info.get("width") or 0),
                height=int(info.get("height") or 0),
                resolution=input_data.resolution,
                seconds=billed_seconds,
                fps=int(round(info.get("fps") or 24)),
                extra={
                    "mode": mode,
                    "model": "pika-h3-flex",
                    "requested_duration": input_data.duration,
                    "measured_seconds": round(float(info.get("seconds") or 0), 3),
                    "charge_micro_usd": charge,
                    "tier": job.get("mode"),
                },
            )] + stem_metas,
        )
        return AppOutput(video=File(path=video_path), stems=stem_files, seed=output.get("seed"), output_meta=output_meta)

    async def on_cancel(self):
        if self.job_id:
            await cancel_job(self.client, self.api_key, self.job_id, self.logger)
        return True

    async def unload(self):
        await self.client.aclose()

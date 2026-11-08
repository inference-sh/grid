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

ENDPOINT = "/v1/media/pika/pika-h3-flex/reference-to-video"

Resolution = Literal["480p", "768p", "1080p", "2k", "4k"]
AspectRatio = Literal["auto", "21:9", "16:9", "3:2", "4:3", "5:4", "1:1", "4:5", "3:4", "2:3", "9:16", "9:21"]


class AppInput(BaseAppInput):
    """Pika H3 Flex reference-to-video: generate a video guided by reference images, videos and audio."""

    prompt: str = Field(
        min_length=1,
        max_length=8000,
        description="What happens in the video. Cite references in order as @Image1, @Image2, @Video1, @Audio1. Describe shots, camera, lighting and sound.",
        examples=["@Image1 walks through the rainy street from @Video1, wearing the jacket in @Image2. The voice from @Audio1 says 'we made it'. Sound: rain, footsteps, no music."],
    )
    reference_images: List[File] = Field(
        default_factory=list, max_length=9,
        description="Up to 9 reference images (PNG, JPEG or WebP, up to 50 MiB each, aspect between 1:4 and 4:1). Cite as @Image1..@Image9.",
    )
    reference_videos: List[File] = Field(
        default_factory=list, max_length=3,
        description="Up to 3 reference videos, 2-15s each and 15s combined. Input video seconds are billed at the output rate. Cite as @Video1..@Video3.",
    )
    reference_audios: List[File] = Field(
        default_factory=list, max_length=3,
        description="Up to 3 reference audio clips, 2-15s each and 15s combined. Free. Cite as @Audio1..@Audio3.",
    )
    duration: int = Field(
        default=5, ge=3, le=15,
        description="Video length in whole seconds, 3 to 15. Billed per output second.",
    )
    aspect_ratio: AspectRatio = Field(
        default="auto",
        description="Output aspect ratio. auto follows the prompt and reference video.",
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

    @model_validator(mode="after")
    def _check_references(self):
        if not self.reference_images and not self.reference_videos:
            raise ValueError("at least one reference image or reference video is required")
        total = len(self.reference_images) + len(self.reference_videos) + len(self.reference_audios)
        if total > 12:
            raise ValueError(f"at most 12 references in total, got {total}")
        return self


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
        self.logger.info("Pika H3 Flex Reference initialized")

    def _build_payload(self, input_data: AppInput, references: List[dict]) -> dict:
        payload = {
            "prompt": input_data.prompt,
            "references": references,
            "duration": input_data.duration,
            "aspect_ratio": input_data.aspect_ratio,
            "resolution": input_data.resolution,
            "seed": input_data.seed,
            "rewrite": input_data.rewrite,
            "output_mode": "audio-stems" if input_data.audio_stems else "video",
        }
        if input_data.audio_stems:
            payload["channel_mode"] = input_data.stems_channel_mode
        return payload

    async def run(self, input_data: AppInput) -> AppOutput:
        self.logger.info(f"References: {len(input_data.reference_images)} images, {len(input_data.reference_videos)} videos, "
                         f"{len(input_data.reference_audios)} audio; resolution: {input_data.resolution}, duration: {input_data.duration}s")

        references: List[dict] = []
        input_metas = []
        for f in input_data.reference_images:
            references.append({"type": "image", "url": await upload_file(self.client, self.api_key, f, "image/png", self.logger)})
            input_metas.append(ImageMeta())
        input_video_seconds = 0.0
        for f in input_data.reference_videos:
            references.append({"type": "video", "url": await upload_file(self.client, self.api_key, f, "video/mp4", self.logger)})
            info = probe_media(f.path) if f.path and os.path.exists(f.path) else {}
            secs = float(info.get("seconds") or 0.0)
            input_video_seconds += secs
            input_metas.append(VideoMeta(width=int(info.get("width") or 0), height=int(info.get("height") or 0), seconds=secs,
                                         fps=int(round(info.get("fps") or 0))))
        for f in input_data.reference_audios:
            references.append({"type": "audio", "url": await upload_file(self.client, self.api_key, f, "audio/mpeg", self.logger)})
            info = probe_media(f.path) if f.path and os.path.exists(f.path) else {}
            input_metas.append(AudioMeta(seconds=float(info.get("seconds") or 0.0), sample_rate=int(info.get("sample_rate") or 0)))
        self.logger.info(f"Input video seconds (billed): {input_video_seconds:.2f}")

        payload = self._build_payload(input_data, references)
        self.job_id = await submit_job(self.client, self.api_key, ENDPOINT, payload, self.logger)
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

        usage = job.get("usage") or {}
        billed_seconds = float(usage.get("output_seconds") or info.get("seconds") or input_data.duration)
        charge = charge_micro_usd(job)

        output_meta = OutputMeta(
            inputs=input_metas,
            outputs=[VideoMeta(
                width=int(info.get("width") or 0),
                height=int(info.get("height") or 0),
                resolution=input_data.resolution,
                seconds=billed_seconds,
                fps=int(round(info.get("fps") or 24)),
                extra={
                    "mode": "reference-to-video",
                    "model": "pika-h3-flex",
                    "requested_duration": input_data.duration,
                    "measured_seconds": round(float(info.get("seconds") or 0), 3),
                    "input_video_seconds": round(input_video_seconds, 3),
                    "charge_micro_usd": charge,
                    "workflow": (job.get("routing") or {}).get("workflow"),
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

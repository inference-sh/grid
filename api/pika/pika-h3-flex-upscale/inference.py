import logging
import os
import tempfile
from typing import Literal, Optional

import httpx
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta
from pydantic import Field

from .pika_helper import (
    PikaError, get_api_key, upload_file, submit_job, poll_job, cancel_job,
    download_file, probe_media, charge_micro_usd,
)

ENDPOINT = "/v1/media/pika/pika-h3-flex/draft-to-video"


class AppInput(BaseAppInput):
    """Upscale a 480p or 768p video to 1080p, 2K or 4K with Pika's H3 Flex upsampler. Frames, frame rate and audio are preserved."""

    video: File = Field(
        description="Source video with a short edge of at most 768 pixels (a 480p or 768p Pika H3 Flex draft, or any similar video). Up to 1441 frames.",
    )
    resolution: Literal["1080p", "2k", "4k"] = Field(
        default="2k",
        description="Target resolution. Billed per output second at the target rate minus the source tier's native rate.",
    )


class AppOutput(BaseAppOutput):
    video: File = Field(description="Upscaled MP4 with the original audio.")


def native_tier(width: int, height: int) -> str:
    short = min(width, height) if width and height else 0
    return "480p" if 0 < short <= 480 else "768p"


class App(BaseApp):

    async def setup(self, metadata):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        self.api_key = get_api_key()
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(900.0, connect=30.0))
        self.job_id: Optional[str] = None
        self.logger.info("Pika H3 Flex Upscale initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        src = probe_media(input_data.video.path) if input_data.video.path and os.path.exists(input_data.video.path) else {}
        sw, sh = int(src.get("width") or 0), int(src.get("height") or 0)
        src_seconds = float(src.get("seconds") or 0.0)
        if sw and sh and min(sw, sh) > 768:
            raise ValueError(f"source short edge is {min(sw, sh)}px; Pika accepts at most 768px (480p or 768p sources)")
        tier = native_tier(sw, sh)
        self.logger.info(f"Source: {sw}x{sh} {src_seconds:.2f}s -> native tier {tier}, target {input_data.resolution}")

        video_url = await upload_file(self.client, self.api_key, input_data.video, "video/mp4", self.logger)
        payload = {"video": video_url, "resolution": input_data.resolution}
        self.job_id = await submit_job(self.client, self.api_key, ENDPOINT, payload, self.logger)
        job = await poll_job(self.client, self.api_key, self.job_id, self.logger)
        self.job_id = None

        output = job.get("output") or {}
        out_url = (output.get("video") or {}).get("url")
        if not out_url:
            raise PikaError(f"completed job has no video url: {str(output)[:300]}")

        workdir = tempfile.mkdtemp(prefix="pika-")
        out_path = await download_file(self.client, out_url, os.path.join(workdir, "video.mp4"), self.logger)
        info = probe_media(out_path)
        self.logger.info(f"Output: {info.get('width')}x{info.get('height')} {info.get('seconds', 0):.2f}s")

        usage = job.get("usage") or {}
        billed_seconds = float(usage.get("output_seconds") or info.get("seconds") or src_seconds)
        output_meta = OutputMeta(
            inputs=[VideoMeta(width=sw, height=sh, resolution=tier, seconds=src_seconds, fps=int(round(src.get("fps") or 0)))],
            outputs=[VideoMeta(
                width=int(info.get("width") or 0),
                height=int(info.get("height") or 0),
                resolution=input_data.resolution,
                seconds=billed_seconds,
                fps=int(round(info.get("fps") or 24)),
                extra={
                    "mode": "upscale",
                    "model": "pika-h3-flex",
                    "source_tier": tier,
                    "measured_seconds": round(float(info.get("seconds") or 0), 3),
                    "charge_micro_usd": charge_micro_usd(job),
                },
            )],
        )
        return AppOutput(video=File(path=out_path), output_meta=output_meta)

    async def on_cancel(self):
        if self.job_id:
            await cancel_job(self.client, self.api_key, self.job_id, self.logger)
        return True

    async def unload(self):
        await self.client.aclose()

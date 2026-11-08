import logging
import os
import tempfile
import zipfile
from typing import List, Literal, Optional

import httpx
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, AudioMeta
from pydantic import Field

from .pika_helper import (
    PikaError, get_api_key, upload_file, submit_job, poll_job, cancel_job,
    download_file, probe_media, charge_micro_usd, stem_seconds,
)

ENDPOINT = "/v1/media/pika/pika-h3-flex/audio-stems"


class AppInput(BaseAppInput):
    """Split the soundtrack of an audio or video file into Dialogue, Music and Effect stems."""

    audio: File = Field(
        description="Audio file, or a video with sound, up to 120 seconds.",
    )
    channel_mode: Literal["mono", "stereo"] = Field(
        default="mono",
        description="mono downmixes before separation; stereo keeps both channels in every stem. Both bill the sum of stem seconds.",
    )


class AppOutput(BaseAppOutput):
    stems: List[File] = Field(default_factory=list, description="Dialogue, Music and Effect WAV files.")
    archive: Optional[File] = Field(default=None, description="ZIP with every stem and the manifest, when Pika returns one.")


class App(BaseApp):

    async def setup(self, metadata):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        self.api_key = get_api_key()
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=30.0))
        self.job_id: Optional[str] = None
        self.logger.info("Pika H3 Flex Audio Stems initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        src = probe_media(input_data.audio.path) if input_data.audio.path and os.path.exists(input_data.audio.path) else {}
        src_seconds = float(src.get("seconds") or 0.0)
        if src_seconds > 120.5:
            raise ValueError(f"input is {src_seconds:.1f}s; Pika accepts at most 120 seconds")
        self.logger.info(f"Input: {src_seconds:.2f}s, channel_mode={input_data.channel_mode}")

        default_type = "video/mp4" if src.get("width") else "audio/mpeg"
        input_url = await upload_file(self.client, self.api_key, input_data.audio, default_type, self.logger)
        payload = {"input_url": input_url, "channel_mode": input_data.channel_mode}
        self.job_id = await submit_job(self.client, self.api_key, ENDPOINT, payload, self.logger)
        job = await poll_job(self.client, self.api_key, self.job_id, self.logger)
        self.job_id = None

        output = job.get("output") or {}
        workdir = tempfile.mkdtemp(prefix="pika-")
        stem_files: List[File] = []
        stem_metas: List[AudioMeta] = []
        archive_file: Optional[File] = None

        archive_url = (output.get("archive") or {}).get("url")
        audio_url = (output.get("audio") or {}).get("url")
        stems = output.get("stems") or []
        if archive_url:
            zip_path = await download_file(self.client, archive_url, os.path.join(workdir, "stems.zip"), self.logger)
            archive_file = File(path=zip_path)
            with zipfile.ZipFile(zip_path) as zf:
                for name in sorted(zf.namelist()):
                    if not name.lower().endswith((".wav", ".flac", ".mp3")) or name.endswith("/"):
                        continue
                    target = os.path.join(workdir, os.path.basename(name))
                    with zf.open(name) as s, open(target, "wb") as d:
                        d.write(s.read())
                    secs = float(probe_media(target).get("seconds") or 0.0)
                    stem_files.append(File(path=target))
                    stem_metas.append(AudioMeta(seconds=secs, extra={"stem": os.path.splitext(os.path.basename(name))[0].lower()}))
        elif stems and any(s.get("url") for s in stems):
            for i, stem in enumerate(stems):
                url = stem.get("url")
                if not url:
                    continue
                name = (stem.get("name") or f"stem_{i + 1}").lower().replace(" ", "_")
                path = await download_file(self.client, url, os.path.join(workdir, f"{name}.wav"), self.logger)
                secs = stem_seconds(stem) or float(probe_media(path).get("seconds") or 0.0)
                stem_files.append(File(path=path))
                stem_metas.append(AudioMeta(seconds=secs, sample_rate=int(stem.get("sample_rate") or 0), extra={"stem": name}))
        elif audio_url:
            path = await download_file(self.client, audio_url, os.path.join(workdir, "stems.bin"), self.logger)
            if zipfile.is_zipfile(path):
                zip_path = os.path.join(workdir, "stems.zip")
                os.replace(path, zip_path)
                archive_file = File(path=zip_path)
                with zipfile.ZipFile(zip_path) as zf:
                    for name in sorted(zf.namelist()):
                        if not name.lower().endswith((".wav", ".flac", ".mp3")) or name.endswith("/"):
                            continue
                        target = os.path.join(workdir, os.path.basename(name))
                        with zf.open(name) as s, open(target, "wb") as d:
                            d.write(s.read())
                        secs = float(probe_media(target).get("seconds") or 0.0)
                        stem_files.append(File(path=target))
                        stem_metas.append(AudioMeta(seconds=secs, extra={"stem": os.path.splitext(os.path.basename(name))[0].lower()}))
            else:
                stem_files.append(File(path=path))
                stem_metas.append(AudioMeta(seconds=float(probe_media(path).get("seconds") or 0.0), extra={"stem": "audio"}))
        else:
            raise PikaError(f"completed job has no stems: {str(output)[:300]}")

        total = sum(m.seconds for m in stem_metas)
        self.logger.info(f"Stems: {[m.extra['stem'] for m in stem_metas]} total {total:.2f}s; charge={charge_micro_usd(job)}")

        output_meta = OutputMeta(
            inputs=[AudioMeta(seconds=src_seconds, sample_rate=int(src.get("sample_rate") or 0))],
            outputs=stem_metas,
        )
        output_meta.outputs[0].extra.update({"model": "pika-h3-flex", "charge_micro_usd": charge_micro_usd(job),
                                             "stem_count": len(stem_metas), "total_stem_seconds": round(total, 3)})
        return AppOutput(stems=stem_files, archive=archive_file, output_meta=output_meta)

    async def on_cancel(self):
        if self.job_id:
            await cancel_job(self.client, self.api_key, self.job_id, self.logger)
        return True

    async def unload(self):
        await self.client.aclose()

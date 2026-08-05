"""
Mirage Video 1 — expressive talking-head video from a portrait image and an audio track.

Given one still image and one speech clip, the model animates the face to match
the voice: lip sync, eye contact, micro-expressions and head motion. You supply
the voice, so any TTS or recorded audio can drive it.

POST /v1/videos -> poll GET /v1/videos/{id} -> GET /v1/videos/{id}/content
"""

import logging
import mimetypes
import os
from typing import Optional

from inferencesh import (
    BaseApp,
    BaseAppInput,
    BaseAppOutput,
    File,
    OutputMeta,
    AudioMeta,
    ImageMeta,
    VideoMeta,
)
from pydantic import Field

from .mirage_helper import (
    download_video_content,
    get_client,
    log_key_fingerprint,
    poll_video,
    post_multipart,
    probe_audio_seconds,
)

MODEL_ID = "mirage-video-1-latest"


class AppInput(BaseAppInput):
    """Input for Mirage Video 1 generation."""

    image: File = Field(
        description=(
            "Portrait image to animate (JPEG or PNG). Use a clear, front-facing shot "
            "of a single person in a close to medium framing, with the mouth open. "
            "Closed mouths and multiple faces in frame both degrade the result."
        )
    )
    audio: File = Field(
        description=(
            "Speech audio that drives the performance (WAV or MP3). The output video "
            "runs as long as this clip. Expressive, natural-sounding audio gives the "
            "best result — audibly synthetic voices look worse on screen."
        )
    )


class AppOutput(BaseAppOutput):
    """Output from Mirage Video 1 generation."""

    video: File = Field(description="The generated talking-head video.")
    video_id: str = Field(
        description="Mirage video ID for this generation, useful when following up on a "
        "specific run with support."
    )


def _upload_tuple(file_obj: File, fallback_type: str) -> tuple:
    path = file_obj.path
    name = file_obj.filename or os.path.basename(path)
    content_type = file_obj.content_type or mimetypes.guess_type(name)[0] or fallback_type
    with open(path, "rb") as fh:
        return (name, fh.read(), content_type)


class App(BaseApp):
    async def setup(self):
        # Without basicConfig the root logger has no handler and every
        # info record is silently dropped instead of reaching task logs.
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        log_key_fingerprint(self.logger)

    async def run(self, input_data: AppInput) -> AppOutput:
        image_path = input_data.image.path
        audio_path = input_data.audio.path
        audio_seconds = probe_audio_seconds(audio_path)
        self.logger.info(
            f"generating with {MODEL_ID}: image={os.path.basename(image_path)} "
            f"audio={os.path.basename(audio_path)} ({audio_seconds:.1f}s)"
        )

        async with get_client() as client:
            created = await post_multipart(
                client,
                "/videos",
                data={"model": MODEL_ID},
                files={
                    "image_reference": _upload_tuple(input_data.image, "image/jpeg"),
                    "audio_reference": _upload_tuple(input_data.audio, "audio/mpeg"),
                },
            )
            video_id = created.get("id") or created["video_id"]
            self.logger.info(f"job created: {video_id} status={created.get('status')}")

            completed = await poll_video(client, video_id)
            video_path = await download_video_content(client, video_id)

        out_meta = VideoMeta.from_file(video_path)
        # A generation that finishes but produces an unreadable file would
        # otherwise be billed as a zero-length video.
        if out_meta.seconds <= 0:
            self.logger.warning(
                "ffprobe reported no duration on the output; "
                f"falling back to the input audio length ({audio_seconds:.1f}s) for metering"
            )
            out_meta.seconds = audio_seconds

        in_metas: list = []
        try:
            from PIL import Image as PILImage

            with PILImage.open(image_path) as im:
                in_metas.append(ImageMeta(width=im.width, height=im.height))
        except Exception:
            self.logger.warning("could not read input image dimensions")
            in_metas.append(ImageMeta())
        in_metas.append(AudioMeta(seconds=audio_seconds))

        self.logger.info(
            f"complete: {out_meta.width}x{out_meta.height} "
            f"{out_meta.seconds:.1f}s @ {out_meta.fps}fps"
        )
        return AppOutput(
            video=File(path=video_path),
            video_id=video_id,
            output_meta=OutputMeta(inputs=in_metas, outputs=[out_meta]),
        )

"""
Mirage Avatar X — expressive talking-head video from a portrait image or
video reference and an audio track.

Provide a still image or a short video clip as the visual reference, plus
a speech audio clip. The model animates the face: lip sync, eye contact,
micro-expressions and head motion. Supply one visual reference — if both
are given, the video reference takes precedence.

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

MODEL_ID = "mirage-avatar-x"

# --- Avatar catalog (Mirage docs list these but the API does not accept
#     an "avatar" form field as of 2026-08-12 — returns 400 missing
#     image_reference. Uncomment and add the field to AppInput + run()
#     when Mirage ships this on the public API.) ---
# PORTRAIT_AVATARS = [
#     "Ayesha", "Farhan", "Giulia", "Jasmine", "Luke",
#     "Maya", "Michael", "Neha", "Tariq", "Valerie",
# ]
# LANDSCAPE_AVATARS = [f"{name} (16:9)" for name in PORTRAIT_AVATARS]
# ALL_AVATARS = PORTRAIT_AVATARS + LANDSCAPE_AVATARS


class AppInput(BaseAppInput):
    """Input for Mirage Avatar X generation."""

    # avatar: Optional[str] = Field(
    #     default=None,
    #     json_schema_extra={"enum": [None] + ALL_AVATARS},
    #     description=(
    #         "Stock avatar from the Mirage catalog. Portrait names produce 9:16 "
    #         "video, names ending in '(16:9)' produce 16:9. "
    #         "Overrides image/video if provided."
    #     ),
    # )
    image: Optional[File] = Field(
        default=None,
        description=(
            "Portrait image (JPEG or PNG, 9:16 or 16:9). Use a clear, front-facing "
            "shot of a single person with mouth open. Ignored when video is provided."
        ),
    )
    video: Optional[File] = Field(
        default=None,
        description=(
            "Video appearance reference (MP4 or MOV, 10–60s). "
            "Takes precedence over image when both are provided."
        ),
    )
    audio: File = Field(
        description=(
            "Speech audio that drives the performance (WAV or MP3, 1–180s). "
            "The output video length matches this clip. Expressive, natural-sounding "
            "audio gives the best result."
        ),
    )


class AppOutput(BaseAppOutput):
    """Output from Mirage Avatar X generation."""

    video: File = Field(description="The generated talking-head video.")
    video_id: str = Field(description="Mirage video ID for this generation.")


def _upload_tuple(file_obj: File, fallback_type: str) -> tuple:
    path = file_obj.path
    name = file_obj.filename or os.path.basename(path)
    content_type = file_obj.content_type or mimetypes.guess_type(name)[0] or fallback_type
    with open(path, "rb") as fh:
        return (name, fh.read(), content_type)


class App(BaseApp):
    async def setup(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        log_key_fingerprint(self.logger)

    async def run(self, input_data: AppInput) -> AppOutput:
        has_video = input_data.video is not None and input_data.video.path
        has_image = input_data.image is not None and input_data.image.path

        if not has_video and not has_image:
            raise ValueError("Provide either an image or a video reference.")

        audio_path = input_data.audio.path
        audio_seconds = probe_audio_seconds(audio_path)

        data = {"model": MODEL_ID}
        files = {"audio_reference": _upload_tuple(input_data.audio, "audio/mpeg")}

        if has_video:
            files["video_reference"] = _upload_tuple(input_data.video, "video/mp4")
            ref_desc = f"video={os.path.basename(input_data.video.path)}"
        else:
            files["image_reference"] = _upload_tuple(input_data.image, "image/jpeg")
            ref_desc = f"image={os.path.basename(input_data.image.path)}"

        self.logger.info(
            f"generating with {MODEL_ID}: {ref_desc} "
            f"audio={os.path.basename(audio_path)} ({audio_seconds:.1f}s)"
        )

        async with get_client() as client:
            created = await post_multipart(
                client,
                "/videos",
                data=data,
                files=files,
            )
            video_id = created.get("id") or created["video_id"]
            self.logger.info(f"job created: {video_id} status={created.get('status')}")

            await poll_video(client, video_id)
            video_path = await download_video_content(client, video_id)

        out_meta = VideoMeta.from_file(video_path)
        if out_meta.seconds <= 0:
            self.logger.warning(
                "ffprobe reported no duration; "
                f"falling back to input audio length ({audio_seconds:.1f}s) for metering"
            )
            out_meta.seconds = audio_seconds

        in_metas: list = [AudioMeta(seconds=audio_seconds)]
        if has_image and not has_video:
            try:
                from PIL import Image as PILImage
                with PILImage.open(input_data.image.path) as im:
                    in_metas.append(ImageMeta(width=im.width, height=im.height))
            except Exception:
                self.logger.warning("could not read input image dimensions")
                in_metas.append(ImageMeta())

        self.logger.info(
            f"complete: {out_meta.width}x{out_meta.height} "
            f"{out_meta.seconds:.1f}s @ {out_meta.fps}fps"
        )
        return AppOutput(
            video=File(path=video_path),
            video_id=video_id,
            output_meta=OutputMeta(inputs=in_metas, outputs=[out_meta]),
        )

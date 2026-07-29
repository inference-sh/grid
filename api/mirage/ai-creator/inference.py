"""
Captions AI Creator — script to avatar video.

Hands a written script to one of Captions' AI Creators (or to an AI Twin you
created with mirage/ai-twin) and returns a finished spokesperson video. Unlike
mirage/video-1 you do not supply audio: the voice is generated from the script.

POST /creator/submit -> poll POST /creator/poll -> download the returned URL
"""

import logging
from typing import List, Literal, Optional

from inferencesh import (
    BaseApp,
    BaseAppInput,
    BaseAppOutput,
    File,
    OutputMeta,
    VideoMeta,
)
from pydantic import Field

from .mirage_helper import (
    download_url,
    get_legacy_client,
    log_key_fingerprint,
    poll_legacy,
    post_legacy,
)

MAX_SCRIPT_CHARS = 800


class AppInput(BaseAppInput):
    """Input for AI Creator video generation."""

    script: str = Field(
        description="What the creator says, up to 800 characters. The voice is generated "
        "from this text.",
        max_length=MAX_SCRIPT_CHARS,
        examples=[
            "I tried this for two weeks and the difference was obvious by day three. "
            "Here's what actually changed."
        ],
    )
    creator_name: str = Field(
        default="Kate",
        description="Name of the AI Creator to perform the script, or the twin_name returned "
        "by mirage/ai-twin. There is no endpoint that lists creators or twins, so this "
        "must be a name you already know.",
    )
    resolution: Literal["fhd", "4k"] = Field(
        default="4k",
        description="Output resolution — fhd is 1080p, 4k is 2160p.",
    )


class AppOutput(BaseAppOutput):
    """Output from AI Creator video generation."""

    video: File = Field(description="The generated creator video.")


RESOLUTION_DIMENSIONS = {"fhd": (1080, 1920), "4k": (2160, 3840)}


class App(BaseApp):
    async def setup(self):
        # Without basicConfig the root logger has no handler and every
        # info record is silently dropped instead of reaching task logs.
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        log_key_fingerprint(self.logger)

    async def run(self, input_data: AppInput) -> AppOutput:
        self.logger.info(
            f"submitting {len(input_data.script)}-char script to creator "
            f"'{input_data.creator_name}' at {input_data.resolution}"
        )

        async with get_legacy_client() as client:
            submitted = await post_legacy(
                client,
                "/creator/submit",
                {
                    "script": input_data.script,
                    "creatorName": input_data.creator_name,
                    "resolution": input_data.resolution,
                },
            )
            operation_id = submitted.get("operationId")
            if not operation_id:
                raise RuntimeError(f"/creator/submit returned no operationId: {submitted}")
            self.logger.info(f"operation {operation_id} submitted, polling")

            result = await poll_legacy(client, "/creator/poll", operation_id)

        url = result.get("url")
        if not url:
            raise RuntimeError(f"operation {operation_id} finished without a video URL: {result}")

        video_path = await download_url(url, suffix=".mp4")
        out_meta = VideoMeta.from_file(video_path)
        if out_meta.width == 0:
            w, h = RESOLUTION_DIMENSIONS[input_data.resolution]
            out_meta.width, out_meta.height = w, h
        out_meta.resolution = "1080p" if input_data.resolution == "fhd" else "4k"

        self.logger.info(
            f"complete: {out_meta.width}x{out_meta.height} {out_meta.seconds:.1f}s"
        )
        return AppOutput(
            video=File(path=video_path),
            # No media inputs — the script is text and billing is on the output video.
            output_meta=OutputMeta(outputs=[out_meta]),
        )

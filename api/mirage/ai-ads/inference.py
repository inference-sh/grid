"""
Captions AI Ads — script plus product media to a UGC-style ad video.

An AI Creator performs the script while your product images and clips are cut in
as B-roll, producing a finished short-form ad. This is the AI Creator flow with
media: the difference from mirage/ai-creator is the mediaUrls the ad is built around.

POST /ads/submit -> poll POST /ads/poll -> download the returned URL
"""

import logging
from typing import List, Literal, Optional

from inferencesh import (
    BaseApp,
    BaseAppInput,
    BaseAppOutput,
    File,
    ImageMeta,
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
    public_url,
)

MAX_SCRIPT_CHARS = 800
MAX_MEDIA = 10

VIDEO_EXTS = (".mp4", ".mov", ".m4v", ".webm")


class AppInput(BaseAppInput):
    """Input for AI Ads video generation."""

    script: str = Field(
        description="Ad script the creator performs, up to 800 characters.",
        max_length=MAX_SCRIPT_CHARS,
        examples=[
            "I was skeptical too. Then I actually used it for a week — and now it's the "
            "first thing I reach for every morning."
        ],
    )
    media: List[File] = Field(
        description=(
            "Product media cut into the ad as B-roll: 1 to 10 files, JPEG, PNG, MOV or MP4. "
            "These must be publicly reachable URLs — the endpoint fetches them by URL "
            "rather than accepting an upload."
        ),
        min_length=1,
        max_length=MAX_MEDIA,
    )
    creator_name: str = Field(
        default="Kate",
        description="Name of the AI Creator to perform the ad, or the twin_name returned by "
        "mirage/ai-twin. There is no endpoint that lists creators or twins, so this must "
        "be a name you already know.",
    )
    resolution: Literal["fhd", "4k"] = Field(
        default="4k",
        description="Output resolution — fhd is 1080p, 4k is 2160p.",
    )


class AppOutput(BaseAppOutput):
    """Output from AI Ads video generation."""

    video: File = Field(description="The generated ad video.")


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
        media_urls = [
            public_url(f, f"media[{i}]") for i, f in enumerate(input_data.media)
        ]
        self.logger.info(
            f"submitting {len(input_data.script)}-char ad script with {len(media_urls)} "
            f"media files to creator '{input_data.creator_name}' at {input_data.resolution}"
        )

        async with get_legacy_client() as client:
            submitted = await post_legacy(
                client,
                "/ads/submit",
                {
                    "script": input_data.script,
                    "creatorName": input_data.creator_name,
                    "mediaUrls": media_urls,
                    "resolution": input_data.resolution,
                },
            )
            operation_id = submitted.get("operationId")
            if not operation_id:
                raise RuntimeError(f"/ads/submit returned no operationId: {submitted}")
            self.logger.info(f"operation {operation_id} submitted, polling")

            result = await poll_legacy(client, "/ads/poll", operation_id)

        url = result.get("url")
        if not url:
            raise RuntimeError(f"operation {operation_id} finished without a video URL: {result}")

        video_path = await download_url(url, suffix=".mp4")
        out_meta = VideoMeta.from_file(video_path)
        if out_meta.width == 0:
            w, h = RESOLUTION_DIMENSIONS[input_data.resolution]
            out_meta.width, out_meta.height = w, h
        out_meta.resolution = "1080p" if input_data.resolution == "fhd" else "4k"

        # Media is referenced by URL and never read locally, so count without
        # dimensions — the count alone is what drives per-input pricing here.
        in_metas: list = []
        for u in media_urls:
            if u.lower().split("?")[0].endswith(VIDEO_EXTS):
                in_metas.append(VideoMeta())
            else:
                in_metas.append(ImageMeta())

        self.logger.info(
            f"complete: {out_meta.width}x{out_meta.height} {out_meta.seconds:.1f}s"
        )
        return AppOutput(
            video=File(path=video_path),
            output_meta=OutputMeta(inputs=in_metas, outputs=[out_meta]),
        )

"""
Mirage Text Overlays — render up to 4 static text variants onto one video.

Upload a clip once and get back a separate video per text variant, each with its
own font, size and colour. Built for hook and headline A/B testing: the same
footage with four different opening lines.

Unlike mirage/video-captions this does not transcribe anything — you supply the
text and it is rendered statically, not timed to speech.

POST /v1/meta/text_overlays -> poll GET /v1/meta/text_overlays/{id}
"""

import logging
import mimetypes
import os
from typing import List, Optional

from inferencesh import (
    BaseApp,
    BaseAppInput,
    BaseAppOutput,
    File,
    OutputMeta,
    VideoMeta,
)
from pydantic import BaseModel, Field, model_validator

from .mirage_helper import (
    download_url,
    get_client,
    log_key_fingerprint,
    poll_text_overlay,
    post_multipart,
)

MAX_TEXTS = 4
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


class AppInput(BaseAppInput):
    """Input for text overlay rendering."""

    video: File = Field(
        description="Source video to render text onto (MP4 or MOV), at most 50 MB."
    )
    texts: List[str] = Field(
        description=(
            "Text variants to render, one output video per entry. Up to 4 — the point is "
            "testing several hooks against the same footage in one call."
        ),
        min_length=1,
        max_length=MAX_TEXTS,
        examples=[["I tried this for a week", "Nobody talks about this", "Day 3 changed it"]],
    )
    fonts: Optional[List[str]] = Field(
        default=None,
        description="Font per variant, aligned by position with texts. Blank entries let "
        "Mirage decide. Omit entirely to auto-pick every font.",
    )
    sizes: Optional[List[str]] = Field(
        default=None,
        description="Font size in pixels per variant, aligned by position with texts. "
        "Blank entries auto-decide.",
    )
    colors: Optional[List[str]] = Field(
        default=None,
        description="Text colour per variant as #RRGGBB, aligned by position with texts. "
        "Blank entries auto-decide.",
    )

    @model_validator(mode="after")
    def _style_lists_align(self):
        # The API aligns these by index, so a shorter list silently shifts styles
        # onto the wrong variants. Reject instead.
        for name in ("fonts", "sizes", "colors"):
            value = getattr(self, name)
            if value is not None and len(value) != len(self.texts):
                raise ValueError(
                    f"'{name}' has {len(value)} entries but 'texts' has {len(self.texts)} — "
                    f"they are matched by position, so the lengths must match. "
                    f"Use an empty string to auto-decide an individual entry."
                )
        return self


class OverlayResult(BaseModel):
    text: str = Field(description="The text variant this result renders")
    video: Optional[File] = Field(
        default=None, description="Rendered video, absent if this variant failed"
    )
    failed: bool = Field(default=False, description="Whether this variant failed")
    error: Optional[str] = Field(default=None, description="Why this variant failed")


class AppOutput(BaseAppOutput):
    """Output from text overlay rendering."""

    videos: List[File] = Field(
        default_factory=list, description="Rendered videos, in the order of texts"
    )
    results: List[OverlayResult] = Field(
        default_factory=list,
        description="Per-variant outcome, including any that failed on their own",
    )
    overlay_id: str = Field(description="Mirage text overlay job ID")


class App(BaseApp):
    async def setup(self):
        # Without basicConfig the root logger has no handler and every
        # info record is silently dropped instead of reaching task logs.
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        log_key_fingerprint(self.logger)

    async def run(self, input_data: AppInput) -> AppOutput:
        path = input_data.video.path
        size = os.path.getsize(path)
        if size > MAX_UPLOAD_BYTES:
            raise RuntimeError(
                f"Video is {size / 1024 / 1024:.1f} MB — the overlay endpoint accepts "
                f"at most 50 MB. Compress it or shorten it first."
            )
        name = input_data.video.filename or os.path.basename(path)
        content_type = (
            input_data.video.content_type or mimetypes.guess_type(name)[0] or "video/mp4"
        )

        # A list value makes httpx emit one form part per entry in order, which is
        # the repeated-field encoding this endpoint expects.
        data: dict = {"texts": list(input_data.texts)}
        for field in ("fonts", "sizes", "colors"):
            values = getattr(input_data, field)
            if values:
                data[field] = list(values)

        self.logger.info(
            f"rendering {len(input_data.texts)} text variant(s) onto {name} "
            f"({size / 1024 / 1024:.1f} MB)"
        )

        async with get_client() as client:
            with open(path, "rb") as fh:
                created = await post_multipart(
                    client,
                    "/meta/text_overlays",
                    data=data,
                    files={"video": (name, fh.read(), content_type)},
                )
            overlay_id = created["id"]
            self.logger.info(f"job created: {overlay_id} status={created.get('status')}")

            completed = await poll_text_overlay(client, overlay_id)

            # A COMPLETE job can still carry failed items, so each is reported
            # rather than collapsed into one success or failure.
            results: List[OverlayResult] = []
            videos: List[File] = []
            for item in sorted(completed.get("results") or [], key=lambda r: r.get("index", 0)):
                text = item.get("text", "")
                if item.get("status") == "COMPLETE" and item.get("video_url"):
                    out_path = await download_url(item["video_url"], suffix=".mp4")
                    f = File(path=out_path)
                    videos.append(f)
                    results.append(OverlayResult(text=text, video=f))
                else:
                    err = item.get("error") or {}
                    message = err.get("message", "no detail")
                    self.logger.warning(f"variant {item.get('index')} failed: {message}")
                    results.append(
                        OverlayResult(text=text, failed=True, error=message)
                    )

        if not videos:
            raise RuntimeError(
                f"All {len(results)} text variant(s) failed: "
                + "; ".join(r.error or "no detail" for r in results)
            )

        out_metas = [VideoMeta.from_file(f.path) for f in videos]
        self.logger.info(f"complete: {len(videos)}/{len(results)} variant(s) rendered")

        return AppOutput(
            videos=videos,
            results=results,
            overlay_id=overlay_id,
            # One upload in, N renders out — the count drives per-output pricing.
            output_meta=OutputMeta(
                inputs=[VideoMeta.from_file(path)],
                outputs=out_metas,
            ),
        )

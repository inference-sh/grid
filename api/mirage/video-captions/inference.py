"""
Mirage Video Captions — burn styled, animated captions onto a vertical video.

Transcribes the video's audio and renders animated captions in one of Mirage's
style templates. Takes an uploaded file only — see the note in run() on why the
endpoint's video_id path is not exposed.

POST /v1/videos/captions -> poll GET /v1/videos/{id} -> GET /v1/videos/{id}/content
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
from pydantic import BaseModel, Field

from .mirage_helper import (
    download_video_content,
    get_client,
    list_caption_templates,
    log_key_fingerprint,
    poll_video,
    post_multipart,
)

# A default so the app is usable without first calling list_templates.
# "Heat" is the template Mirage's own getting-started guide demonstrates.
DEFAULT_TEMPLATE_ID = "ctpl_DxflLOnuKkb198FNdI9E"

MAX_UPLOAD_BYTES = 50 * 1024 * 1024


class AppInput(BaseAppInput):
    """Input for adding captions to a video."""

    video: File = Field(
        description=(
            "Video to caption (MP4 or MOV). Must be 9:16 vertical, at most 50 MB and "
            "5 minutes."
        ),
    )
    caption_template_id: str = Field(
        default=DEFAULT_TEMPLATE_ID,
        description=(
            "Caption style template ID (ctpl_...). Run the list_templates function to see "
            "every available style with a preview video. Defaults to the 'Heat' style."
        ),
    )


class AppOutput(BaseAppOutput):
    """Output from the captioning job."""

    video: File = Field(description="The captioned video.")
    video_id: str = Field(description="Mirage video ID of the captioned result.")


class ListTemplatesInput(BaseAppInput):
    """List the available caption style templates."""

    limit: int = Field(
        default=50, ge=1, le=100, description="Maximum number of templates to return."
    )


class CaptionTemplate(BaseModel):
    id: str = Field(description="Template ID to pass as caption_template_id")
    name: str = Field(description="Human-readable style name")
    preview_url: Optional[str] = Field(
        default=None, description="URL of a short preview video of the style"
    )


class ListTemplatesOutput(BaseAppOutput):
    templates: List[CaptionTemplate] = Field(
        default_factory=list, description="Available caption style templates"
    )


class App(BaseApp):
    async def setup(self):
        # Without basicConfig the root logger has no handler and every
        # info record is silently dropped instead of reaching task logs.
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        log_key_fingerprint(self.logger)

    async def run(self, input_data: AppInput) -> AppOutput:
        # The endpoint also accepts a bare video_id, which is deliberately not
        # exposed: every caller shares one upstream key, so the provider applies
        # no ownership check and any id would be captionable by anyone.
        data = {"caption_template_id": input_data.caption_template_id}

        path = input_data.video.path
        size = os.path.getsize(path)
        if size > MAX_UPLOAD_BYTES:
            raise RuntimeError(
                f"Video is {size / 1024 / 1024:.1f} MB — the captions endpoint accepts "
                f"at most 50 MB. Compress it or shorten it first."
            )
        name = input_data.video.filename or os.path.basename(path)
        content_type = (
            input_data.video.content_type
            or mimetypes.guess_type(name)[0]
            or "video/mp4"
        )
        with open(path, "rb") as fh:
            files = {"video": (name, fh.read(), content_type)}
        self.logger.info(
            f"captioning {name} ({size / 1024 / 1024:.1f} MB) "
            f"with template {input_data.caption_template_id}"
        )
        in_metas: list = [VideoMeta.from_file(path)]

        async with get_client() as client:
            created = await post_multipart(client, "/videos/captions", data=data, files=files)
            video_id = created.get("id") or created["video_id"]
            self.logger.info(f"job created: {video_id} status={created.get('status')}")

            await poll_video(client, video_id)
            out_path = await download_video_content(client, video_id)

        out_meta = VideoMeta.from_file(out_path)

        self.logger.info(
            f"complete: {out_meta.width}x{out_meta.height} {out_meta.seconds:.1f}s"
        )
        return AppOutput(
            video=File(path=out_path),
            video_id=video_id,
            output_meta=OutputMeta(inputs=in_metas, outputs=[out_meta]),
        )

    async def list_templates(self, input_data: ListTemplatesInput) -> ListTemplatesOutput:
        """Fetch the caption style templates available to this account."""
        async with get_client() as client:
            raw = await list_caption_templates(client, limit=input_data.limit)

        self.logger.info(f"found {len(raw)} caption templates")
        return ListTemplatesOutput(
            templates=[
                CaptionTemplate(
                    id=t.get("id", ""),
                    name=t.get("name", ""),
                    preview_url=t.get("preview_url"),
                )
                for t in raw
            ]
        )

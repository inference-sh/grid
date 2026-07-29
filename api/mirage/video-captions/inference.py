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
from typing import List, Literal, Optional

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

# Baked from GET /v1/videos/captions/templates on 2026-07-29 (67 styles) so the
# style picker is a dropdown instead of an opaque ctpl_ string. Mirage adds styles
# over time and this list will drift: `caption_template_id` accepts a raw id as an
# escape hatch, and the list_templates function returns the live set with previews.
CAPTION_TEMPLATES: dict[str, str] = {
    "Acamar": "ctpl_ojEuI2F9lnZ9u91YkYjo",
    "Alcyone": "ctpl_SHr4UDH7uqZPfXVYIlBC",
    "Alhena": "ctpl_3NY2lfJiFAULlIG2COz9",
    "Altair": "ctpl_pwQ0QiBOYuuRvDuEYzmr",
    "Andromeda": "ctpl_7uui8xzgcjbzVVl1WKaE",
    "Aries": "ctpl_pUtOSPltDzsoYJgLBYmo",
    "Arion Pink": "ctpl_Dujj8gkcMe6hoaptLWwU",
    "Baseline": "ctpl_hQisXGK98sAN8E5M4g8h",
    "Betelgeuse": "ctpl_UK5Mjc782KB2qouPo709",
    "Blueprint": "ctpl_X20BklSG9zplCCfMrZJI",
    "Buzz": "ctpl_yvE0ZnYzEj6ClCD2ee1f",
    "Cartwheel Black": "ctpl_xCFRVbYyA4OShvj3jPA8",
    "Cartwheel Purple": "ctpl_Ce9huKC7BtvGK6tQcoXX",
    "Castor": "ctpl_Av1JgOg6DoJlXJLwfFFm",
    "Chronicle": "ctpl_Ck8XVyVLYN2YRzwzzwxm",
    "Closed Cap": "ctpl_fYTszWbhnFlgJp2cU4vI",
    "Cove": "ctpl_FbiwB6xaJm9CQA3Ot4Tx",
    "Cygnus A": "ctpl_JtJChccdjhEQEC1wiQcn",
    "Daily Mail": "ctpl_9YoEtziK8O9WvuWYzNZ7",
    "Dimidium": "ctpl_tWgDpYUXw4wyU7tB7eOG",
    "Doodle": "ctpl_NBpLQoUt2a6hRPdSD3Qp",
    "Drive": "ctpl_wR9PXfmxW1DFxEUuATFg",
    "Eclipse": "ctpl_hVijUCESRcnAG1TqV8Bv",
    "Energy": "ctpl_oofP3mxbx8CaEPNYqnKD",
    "Energy II": "ctpl_iusqRnf5W08ENWPOVvkz",
    "Finlay": "ctpl_iqX03uuwVIqPJOV4MGgr",
    "Flair": "ctpl_SP9YZolOKiJv8tJqMeYC",
    "Footprint v3": "ctpl_XZwWZdooFDLbmiZ524p5",
    "Freshly": "ctpl_SprKAix4SRgIUFwpxOZ1",
    "Fuel": "ctpl_17zhen9AkxDAtWNL67ir",
    "Garnet": "ctpl_2ryQmZz2Iq4XRX4F8pCU",
    "Glow": "ctpl_P1qdlnbSk4NZERdu6tIO",
    "Growth": "ctpl_A503B1LAyB7A4U2x84p7",
    "Heat": "ctpl_DxflLOnuKkb198FNdI9E",
    "Helios": "ctpl_lXEe3rYgxCKh2MnCymqC",
    "Hustle v3": "ctpl_epPY6aFemTA34RuDT9yv",
    "Linear": "ctpl_Fy5WxiGPPAV393kth8mZ",
    "Lumin": "ctpl_J6YuQpoLwYlBVgSKTIrw",
    "Magazine": "ctpl_vrs1M2VrxvzQWNRypRvh",
    "Marigold": "ctpl_fsR2Jc3zhDrsMEOPb9mo",
    "Medusa": "ctpl_yNnJyDLSH5oIouKdjQx2",
    "Messages": "ctpl_UoxfGUNJyd21EOr8kClC",
    "Milky Way": "ctpl_jcTmJGX77Uwz2AqLOX4S",
    "Million": "ctpl_A1AMNbzmIat0CEiwrgfI",
    "Minima": "ctpl_grdfFHRFNm6sBARtdVRy",
    "Mizar": "ctpl_idIjDO4Mtwu9nquVB0OV",
    "Monster": "ctpl_tkuSt0SUnBuxNT6b2LNG",
    "Neon": "ctpl_hfkGwYGSIPHijM4vWRqH",
    "Note": "ctpl_iV3lX880qcCe9AURc4XA",
    "Nova": "ctpl_hMydc4rk9l4yC3Hw6MKK",
    "Orbitar Black": "ctpl_NCLdW43y7fggCYnS5miH",
    "Orion": "ctpl_JJFaDOxMmHWj5B5qSklN",
    "Pacific": "ctpl_bKvIxXSn2sZvmt7RbFV1",
    "Poem": "ctpl_5RaXhC2spDHYw40DsgFF",
    "Pollux": "ctpl_rn0HysnZUu5UzZEJw8gq",
    "Pulse": "ctpl_ZLUhoYk5omnYGzcKY1aQ",
    "Recess": "ctpl_slCGoQERGj5Dn9Cr1Whd",
    "Runway": "ctpl_NVTtZ8SY0Jo4UUFokB8p",
    "Scene": "ctpl_tN68l72WH2RFjl1eMhwg",
    "Script": "ctpl_2sOSSKWNXgQu3C5eE60l",
    "Sirius": "ctpl_miZu2nLWyP7X8oEAAHcM",
    "Suzy": "ctpl_DcsGeQFyiKLSqHAfC5fF",
    "Techwave": "ctpl_3PwgOIKd3tbOstKPPC4E",
    "Thuban": "ctpl_LWMH1tfwvAHpzq0LAK7V",
    "Vitamin B": "ctpl_RbfrMonqCaUZbIWZGlG4",
    "Vitamin C": "ctpl_qdtwV5Vi2GbkQZ9THLcW",
    "Zodiac": "ctpl_7ukpFvJbH1PplZLpCz8t",
}

# "Heat" is the style Mirage's own getting-started guide demonstrates.
DEFAULT_STYLE = "Heat"

MAX_UPLOAD_BYTES = 50 * 1024 * 1024


CaptionStyle = Literal[tuple(CAPTION_TEMPLATES)]  # type: ignore[misc]


class AppInput(BaseAppInput):
    """Input for adding captions to a video."""

    video: File = Field(
        description=(
            "Video to caption (MP4 or MOV). Must be 9:16 vertical, at most 50 MB and "
            "5 minutes."
        ),
    )
    caption_style: CaptionStyle = Field(
        default=DEFAULT_STYLE,
        description=(
            "Caption style to render. Run the list_templates function to preview each one "
            "as a short video."
        ),
    )
    caption_template_id: Optional[str] = Field(
        default=None,
        description=(
            "Raw template id (ctpl_...), overriding caption_style. Only needed for a style "
            "newer than this app's baked list — check list_templates for ids."
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
        # A raw id wins over the picker, so a style newer than the baked list is
        # still reachable without a redeploy.
        template_id = input_data.caption_template_id or CAPTION_TEMPLATES[input_data.caption_style]

        # The endpoint also accepts a bare video_id, which is deliberately not
        # exposed: every caller shares one upstream key, so the provider applies
        # no ownership check and any id would be captionable by anyone.
        data = {"caption_template_id": template_id}

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
            f"with style {input_data.caption_style} ({template_id})"
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

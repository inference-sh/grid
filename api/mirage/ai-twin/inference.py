"""
Captions AI Twin — build a reusable avatar from your own footage.

Give it a calibration video of a person speaking plus a handful of calibration
stills, and it trains an AI Twin: a named avatar with that person's likeness and
voice. Once created, the twin's name can be passed as creator_name to
mirage/ai-creator (script to video) and mirage/ai-ads.

This app returns the twin's name and operation ID, not a video — the video comes
later from whichever app performs a script with it.

POST /twin/create -> poll POST /twin/status
"""

import logging
from typing import List, Optional

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
    get_legacy_client,
    log_key_fingerprint,
    poll_legacy,
    post_legacy,
    public_url,
    scoped_name,
)


class AppInput(BaseAppInput):
    """Input for AI Twin creation."""

    label: str = Field(
        description="A label for the twin, for your own reference. The twin's actual "
        "name is this label prefixed with your team and task id — use the twin_name "
        "returned by this app as creator_name in mirage/ai-creator or mirage/ai-ads. "
        "Record it: there is no endpoint that lists your twins.",
        examples=["my-spokesperson"],
    )
    calibration_video: File = Field(
        description=(
            "Calibration video of the person speaking to camera. Must be a publicly "
            "reachable URL — the endpoint fetches media by URL rather than accepting "
            "an upload."
        )
    )
    calibration_images: List[File] = Field(
        description="Calibration stills of the same person, as publicly reachable URLs.",
        min_length=1,
    )
    language: str = Field(
        default="English",
        description="Language spoken in the calibration video.",
    )


class AppOutput(BaseAppOutput):
    """Result of AI Twin creation."""

    twin_name: str = Field(
        description="The twin's full name — pass this as creator_name to mirage/ai-creator "
        "or mirage/ai-ads to generate video with it. Save it; it cannot be looked up later."
    )
    operation_id: str = Field(description="Captions operation ID for the creation job.")
    state: str = Field(description="Final state reported by /twin/status.")


class App(BaseApp):
    async def setup(self):
        # Without basicConfig the root logger has no handler and every
        # info record is silently dropped instead of reaching task logs.
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        log_key_fingerprint(self.logger)

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        # Derived per request: this worker also serves other tenants.
        twin_name = scoped_name(metadata, input_data.label, log=self.logger)

        video_url = public_url(input_data.calibration_video, "calibration_video")
        image_urls = [
            public_url(f, f"calibration_images[{i}]")
            for i, f in enumerate(input_data.calibration_images)
        ]
        self.logger.info(
            f"creating twin '{twin_name}' ({input_data.language}) from 1 video "
            f"and {len(image_urls)} calibration images"
        )

        async with get_legacy_client() as client:
            submitted = await post_legacy(
                client,
                "/twin/create",
                {
                    "name": twin_name,
                    "videoUrl": video_url,
                    "calibrationImageUrls": image_urls,
                    "language": input_data.language,
                },
            )
            operation_id = submitted.get("operationId")
            if not operation_id:
                raise RuntimeError(f"/twin/create returned no operationId: {submitted}")
            self.logger.info(f"operation {operation_id} submitted, polling")

            result = await poll_legacy(client, "/twin/status", operation_id)

        state = result.get("state", "COMPLETE")
        self.logger.info(f"twin '{twin_name}' finished in state {state}")

        return AppOutput(
            twin_name=twin_name,
            operation_id=operation_id,
            state=state,
            # Training consumes the calibration media; there is no media output.
            output_meta=OutputMeta(
                inputs=[VideoMeta()] + [ImageMeta() for _ in image_urls],
                outputs=[],
            ),
        )

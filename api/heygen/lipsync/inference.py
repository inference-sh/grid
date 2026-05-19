"""
HeyGen Lipsync - Re-sync video lip movements to new audio.

Takes an existing video and new audio, re-generates lip movements
to match the audio with natural-looking results.
"""

import logging
from typing import Optional, Literal

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta
from pydantic import Field

from .heygen_helper import (
    get_client,
    post_endpoint,
    poll_lipsync,
    download_file,
    build_asset_ref,
)


ModeType = Literal["speed", "precision"]


class AppInput(BaseAppInput):
    """Input for lipsync generation."""

    video: File = Field(
        description="Source video to re-sync. Must be a publicly accessible URL.",
    )
    audio: File = Field(
        description="Audio file to sync lips to. Must be a publicly accessible URL.",
    )
    title: Optional[str] = Field(
        default=None,
        description="Title for the lipsync job.",
    )
    mode: ModeType = Field(
        default="speed",
        description="Lipsync mode. 'speed' is faster, 'precision' is higher quality.",
    )
    enable_caption: bool = Field(
        default=False,
        description="Generate captions for the output video.",
    )
    enable_dynamic_duration: bool = Field(
        default=True,
        description="Allow dynamic duration adjustment.",
    )
    disable_music_track: bool = Field(
        default=False,
        description="Remove background music from the output.",
    )
    start_time: Optional[float] = Field(
        default=None,
        description="Start time in seconds for partial lipsync.",
    )
    end_time: Optional[float] = Field(
        default=None,
        description="End time in seconds for partial lipsync.",
    )


class AppOutput(BaseAppOutput):
    """Output from lipsync generation."""

    video: File = Field(description="The lip-synced video.")


class App(BaseApp):
    async def setup(self):
        self.logger = logging.getLogger(__name__)

    async def run(self, input_data: AppInput) -> AppOutput:
        self.logger.info("Starting lipsync...")

        video_ref = build_asset_ref(input_data.video)
        audio_ref = build_asset_ref(input_data.audio)

        payload = {
            "video": video_ref,
            "audio": audio_ref,
            "mode": input_data.mode,
            "enable_caption": input_data.enable_caption,
            "enable_dynamic_duration": input_data.enable_dynamic_duration,
            "disable_music_track": input_data.disable_music_track,
        }

        if input_data.title:
            payload["title"] = input_data.title
        if input_data.start_time is not None:
            payload["start_time"] = input_data.start_time
        if input_data.end_time is not None:
            payload["end_time"] = input_data.end_time

        async with get_client(timeout=300) as client:
            result = await post_endpoint(client, "/v3/lipsyncs", payload)

            lipsync_id = result.get("lipsync_id")
            if not lipsync_id:
                raise RuntimeError("No lipsync_id returned from HeyGen API")

            self.logger.info(f"Lipsync {lipsync_id} created, polling...")
            completed = await poll_lipsync(client, lipsync_id)

        video_url = completed.get("video_url")
        if not video_url:
            raise RuntimeError("No video URL in completed lipsync")

        video_path = await download_file(video_url)
        duration = completed.get("duration", 0)

        output_meta = OutputMeta(
            outputs=[
                VideoMeta(
                    width=0,
                    height=0,
                    seconds=float(duration),
                    fps=24,
                    extra={"mode": input_data.mode},
                )
            ]
        )

        self.logger.info(f"Lipsync complete: {duration}s")
        return AppOutput(video=File(path=video_path), output_meta=output_meta)

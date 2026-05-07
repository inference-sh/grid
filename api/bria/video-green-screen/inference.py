from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta
from pydantic import Field
from typing import Optional, Literal
import logging

from . import bria_helper

logger = logging.getLogger(__name__)

OUTPUT_CODECS = Literal[
    "mp4_h264", "mp4_h265", "webm_vp9", "mov_h265",
    "mov_proresks", "mkv_h264", "mkv_h265", "mkv_vp9", "gif",
]


class AppInput(BaseAppInput):
    video_url: str = Field(description="URL of the input video (MP4, MOV, WebM, AVI, GIF). Max 60s, up to 16K.")
    green_shade: Optional[Literal["broadcast_green", "chroma_green", "blue_screen"]] = Field(
        default=None, description="Screen color: broadcast_green (#00B140), chroma_green (#00FF00), or blue_screen (#0000FF). Default: broadcast_green."
    )
    output_container_and_codec: Optional[OUTPUT_CODECS] = Field(
        default=None, description="Output format preset (default: mp4_h264)"
    )
    preserve_audio: Optional[bool] = Field(default=None, description="Retain audio track from input video (default: true)")


class AppOutput(BaseAppOutput):
    video: File = Field(description="Video with green/blue screen background applied")


class App(BaseApp):
    async def setup(self, config):
        self.client = bria_helper.get_client(timeout=300)
        logger.info("Bria Video Green Screen ready")

    async def run(self, input_data: AppInput) -> AppOutput:
        payload = {"video": input_data.video_url}
        if input_data.green_shade is not None:
            payload["green_shade"] = input_data.green_shade
        if input_data.output_container_and_codec is not None:
            payload["output_container_and_codec"] = input_data.output_container_and_codec
        if input_data.preserve_audio is not None:
            payload["preserve_audio"] = input_data.preserve_audio

        logger.info("Requesting video green screen effect")
        result = await bria_helper.call_endpoint(
            self.client, "green_screen", payload,
            base_url=bria_helper.VIDEO_EDIT_URL,
        )

        video_url = result["result"]["video_url"]
        path = await bria_helper.download_video(self.client, video_url)
        logger.info(f"Downloaded green screen video to {path}")

        return AppOutput(
            video=File(path=path),
            output_meta=OutputMeta(outputs=[VideoMeta(duration_seconds=0, width=0, height=0, count=1)]),
        )

    async def unload(self):
        await self.client.aclose()

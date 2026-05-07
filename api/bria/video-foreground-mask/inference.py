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
    output_container_and_codec: Optional[OUTPUT_CODECS] = Field(
        default=None, description="Output format preset (default: mp4_h264)"
    )


class AppOutput(BaseAppOutput):
    video: File = Field(description="Foreground segmentation mask video")


class App(BaseApp):
    async def setup(self, config):
        self.client = bria_helper.get_client(timeout=300)
        logger.info("Bria Video Foreground Mask ready")

    async def run(self, input_data: AppInput) -> AppOutput:
        payload = {"video": input_data.video_url}
        if input_data.output_container_and_codec is not None:
            payload["output_container_and_codec"] = input_data.output_container_and_codec

        logger.info("Requesting video foreground mask")
        result = await bria_helper.call_endpoint(
            self.client, "foreground_mask", payload,
            base_url="https://engine.prod.bria-api.com/v1/video",
        )

        video_url = result["result"]["video_url"]
        path = await bria_helper.download_video(self.client, video_url)
        logger.info(f"Downloaded foreground mask video to {path}")

        return AppOutput(
            video=File(path=path),
            output_meta=OutputMeta(outputs=[VideoMeta(duration_seconds=0, width=0, height=0, count=1)]),
        )

    async def unload(self):
        await self.client.aclose()

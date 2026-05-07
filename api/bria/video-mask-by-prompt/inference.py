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
    video_url: str = Field(description="URL of the input video (MP4, MOV, WebM, AVI, GIF). Max 5s, 750p max.")
    prompt: str = Field(description="Text prompt describing the object to segment and mask")
    output_container_and_codec: Optional[OUTPUT_CODECS] = Field(
        default=None, description="Output format preset (default: mp4_h264)"
    )


class AppOutput(BaseAppOutput):
    video: File = Field(description="Segmentation mask video generated from text prompt")


class App(BaseApp):
    async def setup(self, config):
        self.client = bria_helper.get_client(timeout=300)
        logger.info("Bria Video Mask by Prompt ready")

    async def run(self, input_data: AppInput) -> AppOutput:
        payload = {
            "video": input_data.video_url,
            "prompt": input_data.prompt,
        }
        if input_data.output_container_and_codec is not None:
            payload["output_container_and_codec"] = input_data.output_container_and_codec

        logger.info(f"Requesting video mask by prompt: {input_data.prompt}")
        result = await bria_helper.call_endpoint(
            self.client, "mask_by_prompt", payload,
            base_url="https://engine.prod.bria-api.com/v1/video/edit",
        )

        video_url = result["result"]["video_url"]
        path = await bria_helper.download_video(self.client, video_url)
        logger.info(f"Downloaded mask video to {path}")

        return AppOutput(
            video=File(path=path),
            output_meta=OutputMeta(outputs=[VideoMeta(duration_seconds=0, width=0, height=0, count=1)]),
        )

    async def unload(self):
        await self.client.aclose()

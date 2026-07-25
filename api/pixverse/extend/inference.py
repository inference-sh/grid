import logging
from typing import Optional

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta
from pydantic import Field

from .pixverse_helper import get_client, upload_video, poll_video, download_file, api_post


class AppInput(BaseAppInput):
    video: File = Field(
        description="Source video to extend. MP4/MOV, max 50MB, max 1920px, max 30s.",
    )
    prompt: str = Field(
        description="Prompt describing what should happen in the extended portion (max 2048 chars)",
    )
    quality: str = Field(
        default="720p",
        description="Output resolution",
        json_schema_extra={"enum": ["360p", "540p", "720p", "1080p"]},
    )
    duration: int = Field(
        default=5,
        description="Extension duration in seconds. 1080p supports only 5s.",
        json_schema_extra={"enum": [5, 8]},
    )
    negative_prompt: Optional[str] = Field(
        default=None,
        description="What to avoid (max 2048 chars)",
    )
    seed: Optional[int] = Field(
        default=None,
        ge=0,
        le=2147483647,
        description="Seed for reproducibility",
    )
    motion_mode: str = Field(
        default="normal",
        description="Motion intensity. Fast mode only available for 5s, not at 1080p.",
        json_schema_extra={"enum": ["normal", "fast"]},
    )


class AppOutput(BaseAppOutput):
    video: File = Field(description="Extended video file")


class App(BaseApp):
    async def setup(self, metadata):
        self.logger = logging.getLogger(__name__)
        self.client = get_client()
        self.cancel_flag = False
        self.logger.info("PixVerse Extend initialized")

    async def on_cancel(self):
        self.cancel_flag = True
        return True

    async def run(self, input_data: AppInput) -> AppOutput:
        self.cancel_flag = False

        video_media_id = await upload_video(self.client, input_data.video.path)

        payload = {
            "video_media_id": video_media_id,
            "model": "v6",
            "prompt": input_data.prompt,
            "quality": input_data.quality,
            "duration": input_data.duration,
            "motion_mode": input_data.motion_mode,
        }
        if input_data.negative_prompt:
            payload["negative_prompt"] = input_data.negative_prompt
        if input_data.seed is not None:
            payload["seed"] = input_data.seed

        self.logger.info(f"Extending video: quality={input_data.quality}, duration={input_data.duration}s")
        resp = await api_post(self.client, "/openapi/v2/video/extend/generate", payload)

        video_id = resp["video_id"]
        self.logger.info(f"Task created: video_id={video_id}")

        result = await poll_video(self.client, video_id)
        video_url = result["url"]
        width = result.get("outputWidth", 0)
        height = result.get("outputHeight", 0)
        self.logger.info(f"Video ready: {width}x{height}")

        video_path = await download_file(video_url)

        return AppOutput(
            video=File(path=video_path),
            output_meta=OutputMeta(
                inputs=[VideoMeta()],
                outputs=[VideoMeta(
                    width=width,
                    height=height,
                    resolution=input_data.quality,
                    seconds=float(input_data.duration),
                    extra={"model": "v6", "mode": "extend"},
                )],
            ),
        )

    async def unload(self):
        await self.client.aclose()

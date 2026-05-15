"""
Kling Video-to-Audio - Add Sound to Any Video

Generate and add appropriate audio (sound effects, ambient, music) to videos.
Works with Kling-generated videos and user-uploaded videos.
"""

import os
import logging
from typing import Optional

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta
from pydantic import Field

from .kling_helper import (
    KlingClient,
    KlingAPIError,
    poll_task,
)
from .download_helper import download_video


class AppInput(BaseAppInput):
    """Kling Video-to-Audio.

    Provide a video (3-20s) and optionally describe the audio you want.
    """

    video: File = Field(
        description="Video to add audio to. 3-20 seconds duration. Formats: mp4, mov.",
    )
    prompt: Optional[str] = Field(
        default=None,
        description="Description of desired audio. E.g. 'Ocean waves, seagulls, gentle wind' or 'Upbeat electronic background music'.",
    )


class AppOutput(BaseAppOutput):
    video: File = Field(description="The video with generated audio.")


class App(BaseApp):

    async def setup(self, metadata):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        logging.getLogger("httpx").setLevel(logging.WARNING)

        access_key = os.environ.get("KLING_ACCESS_KEY")
        secret_key = os.environ.get("KLING_SECRET_KEY")
        if not access_key or not secret_key:
            raise RuntimeError("KLING_ACCESS_KEY and KLING_SECRET_KEY must be set")

        self.client = KlingClient(access_key=access_key, secret_key=secret_key)
        self.logger.info("Kling Video-to-Audio initialized")

    async def on_cancel(self):
        self.logger.info("Cancellation requested")
        return True

    async def run(self, input_data: AppInput) -> AppOutput:
        self.logger.info(f"Adding audio to video, prompt: {input_data.prompt[:100] if input_data.prompt else 'auto'}")

        task = await self.client.video_to_audio.create(
            video_url=input_data.video.uri,
            prompt=input_data.prompt,
        )

        self.logger.info(f"Task created: {task.task_id}")

        result = await poll_task(
            self.client.video_to_audio.get,
            task.task_id,
            interval=3.0,
            timeout=300.0,
        )

        if not result.videos or not result.videos[0].url:
            raise RuntimeError(f"No video URL in result: {result.task_status_msg}")

        video_url = result.videos[0].url
        video_duration = float(result.videos[0].duration) if result.videos[0].duration else 5.0
        self.logger.info(f"Video ready: {video_url[:80]}..., duration={video_duration}s")

        video_path = download_video(video_url, self.logger)

        output_meta = OutputMeta(
            outputs=[
                VideoMeta(
                    seconds=video_duration,
                    extra={"model": "kling-video-to-audio"},
                )
            ]
        )

        return AppOutput(video=File(path=video_path), output_meta=output_meta)

    async def unload(self):
        await self.client.close()

"""
Kling Lip Sync - Drive Mouth Movements with Text or Audio

Modify existing videos to make characters' lips match new audio or text.
Ideal for dubbing, adding speech to silent videos, or replacing dialogue.
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
    """Kling Lip Sync.

    Provide a video and either text+voice_id or audio to drive lip movements.
    """

    video: File = Field(
        description="Source video with a visible face. Formats: mp4, mov.",
    )
    text: Optional[str] = Field(
        default=None,
        description="Text for the character to speak. Requires voice_id.",
    )
    audio: Optional[File] = Field(
        default=None,
        description="Audio file to drive lip sync. Alternative to text+voice_id.",
    )
    voice_id: Optional[str] = Field(
        default=None,
        description="TTS voice ID. Required when using text input.",
    )


class AppOutput(BaseAppOutput):
    video: File = Field(description="The lip-synced video file.")


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
        self.logger.info("Kling Lip Sync initialized")

    async def on_cancel(self):
        self.logger.info("Cancellation requested")
        return True

    async def run(self, input_data: AppInput) -> AppOutput:
        if not input_data.text and not input_data.audio:
            raise RuntimeError("Either text+voice_id or audio must be provided")
        if input_data.text and not input_data.voice_id:
            raise RuntimeError("voice_id is required when using text input")

        mode = "text" if input_data.text else "audio"
        self.logger.info(f"Mode: {mode}-driven lip sync")

        task = await self.client.lip_sync.create(
            video_url=input_data.video.uri,
            text=input_data.text,
            audio_url=input_data.audio.uri if input_data.audio else None,
            voice_id=input_data.voice_id,
        )

        self.logger.info(f"Task created: {task.task_id}")

        result = await poll_task(
            self.client.lip_sync.get,
            task.task_id,
            interval=3.0,
            timeout=600.0,
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
                    extra={"mode": mode, "model": "kling-lip-sync"},
                )
            ]
        )

        return AppOutput(video=File(path=video_path), output_meta=output_meta)

    async def unload(self):
        await self.client.close()

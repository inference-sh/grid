"""
Kling Lip Sync - Drive Mouth Movements with Audio

Advanced lip-sync: identifies faces in a video, then drives mouth movements
with provided audio. Two-step process handled automatically.
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

    Provide a video with a visible face and an audio file.
    The app automatically identifies faces and applies lip-sync.
    """

    video: File = Field(
        description="Source video with a visible face. Formats: mp4, mov.",
    )
    audio: File = Field(
        description="Audio file to drive lip sync. Formats: mp3, wav, m4a, aac. Max 5MB, 2-60 seconds.",
    )
    sound_volume: float = Field(
        default=2.0,
        ge=0.0,
        le=2.0,
        description="Volume of the lip-sync audio (0-2, default 2).",
    )
    original_audio_volume: float = Field(
        default=2.0,
        ge=0.0,
        le=2.0,
        description="Volume of the original video audio (0-2, default 2). Set to 0 to mute original.",
    )
    sound_end_time: Optional[int] = Field(
        default=None,
        description="Audio crop end point in milliseconds. The cropped audio must not exceed video duration. If not set, uses the video duration.",
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
        # Step 1: Identify faces in the video
        self.logger.info("Step 1: Identifying faces in video...")
        face_data = await self.client.lip_sync.identify_face(input_data.video.uri)

        session_id = face_data.get("session_id", "")
        faces = face_data.get("face_data", []) or face_data.get("face_list", [])
        if not session_id or not faces:
            raise RuntimeError(f"No faces detected in video. Response: {face_data}")

        face_id = faces[0].get("face_id", "0")
        face_end = faces[0].get("end_time", 5000)
        self.logger.info(f"Found {len(faces)} face(s), face_id={face_id}, session={session_id}, face_end={face_end}ms")

        # Step 2: Create lip-sync task
        # sound_end_time = how much of the audio to use (ms from audio start)
        # The cropped audio length must not exceed video duration
        sound_end = input_data.sound_end_time if input_data.sound_end_time else face_end
        self.logger.info(f"Step 2: Creating lip-sync task, sound_end_time={sound_end}ms")

        task = await self.client.lip_sync.create(
            session_id=session_id,
            face_id=face_id,
            sound_file=input_data.audio.uri,
            sound_end_time=sound_end,
            sound_volume=input_data.sound_volume,
            original_audio_volume=input_data.original_audio_volume,
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
                    extra={"model": "kling-advanced-lip-sync"},
                )
            ]
        )

        return AppOutput(video=File(path=video_path), output_meta=output_meta)

    async def unload(self):
        await self.client.close()

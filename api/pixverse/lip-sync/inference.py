import logging
from typing import Optional

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta, RawMeta
from pydantic import Field

from .pixverse_helper import get_client, upload_video, upload_audio, poll_video, download_file, api_post


class AppInput(BaseAppInput):
    video: File = Field(
        description="Source video for lip sync. MP4/MOV/WebM, max 100MB, max 1920px, max 60s.",
    )
    audio: Optional[File] = Field(
        default=None,
        description="Audio file to sync. MP3/WAV/M4A/AAC, max 60s. Provide either audio or tts_text.",
    )
    tts_text: Optional[str] = Field(
        default=None,
        description="Text for TTS lip sync (max 200 chars). Provide either tts_text or audio.",
    )
    tts_speaker_id: Optional[str] = Field(
        default=None,
        description="TTS voice ID. Use 'auto' for automatic selection. Required when using tts_text.",
    )


class AppOutput(BaseAppOutput):
    video: File = Field(description="Video with lip sync applied")


class App(BaseApp):
    async def setup(self, metadata):
        self.logger = logging.getLogger(__name__)
        self.client = get_client()
        self.cancel_flag = False
        self.logger.info("PixVerse Lip Sync initialized")

    async def on_cancel(self):
        self.cancel_flag = True
        return True

    async def run(self, input_data: AppInput) -> AppOutput:
        self.cancel_flag = False

        if not input_data.audio and not input_data.tts_text:
            raise ValueError("Provide either audio file or tts_text for lip sync")

        video_media_id = await upload_video(self.client, input_data.video.path)

        payload = {"video_media_id": video_media_id}

        if input_data.audio:
            self.logger.info("Audio lip sync mode")
            audio_media_id = await upload_audio(self.client, input_data.audio.path)
            payload["audio_media_id"] = audio_media_id
        else:
            self.logger.info("TTS lip sync mode")
            payload["lip_sync_tts_content"] = input_data.tts_text
            payload["lip_sync_tts_speaker_id"] = input_data.tts_speaker_id or "auto"

        self.logger.info("Creating lip sync task")
        resp = await api_post(self.client, "/openapi/v2/video/lip_sync/generate", payload)

        video_id = resp["video_id"]
        credits_used = resp.get("credits", 0)
        self.logger.info(f"Task created: video_id={video_id}, credits={credits_used}")

        result = await poll_video(self.client, video_id)
        video_url = result["url"]
        width = result.get("outputWidth", 0)
        height = result.get("outputHeight", 0)
        self.logger.info(f"Video ready: {width}x{height}")

        video_path = await download_file(video_url)

        outputs = [VideoMeta(
            width=width,
            height=height,
            extra={"model": "lip-sync"},
        )]
        if credits_used:
            outputs.append(RawMeta(cost=credits_used))

        return AppOutput(
            video=File(path=video_path),
            output_meta=OutputMeta(
                inputs=[VideoMeta()],
                outputs=outputs,
            ),
        )

    async def unload(self):
        await self.client.aclose()

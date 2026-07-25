import logging
from typing import Optional

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta, ImageMeta, RawMeta
from pydantic import Field

from .pixverse_helper import get_client, upload_image, upload_audio, poll_video, download_file, api_post


class AppInput(BaseAppInput):
    image: File = Field(
        description="Portrait image for avatar. PNG/JPG/JPEG/WebP, max 20MB, max 10000px.",
    )
    audio: Optional[File] = Field(
        default=None,
        description="Audio file for avatar speech. MP3/WAV/M4A/AAC, max 30s. Provide either audio or tts_text.",
    )
    tts_text: Optional[str] = Field(
        default=None,
        description="Text for TTS avatar speech (30-200 chars). Provide either tts_text or audio.",
    )
    tts_speaker_id: Optional[str] = Field(
        default=None,
        description="TTS voice ID. Use 'auto' for automatic selection. Required when using tts_text.",
    )
    quality: str = Field(
        default="720p",
        description="Output video resolution",
        json_schema_extra={"enum": ["720p", "1080p"]},
    )
    prompt: Optional[str] = Field(
        default=None,
        description="Visual guidance for avatar movements, pose, and expressions",
    )


class AppOutput(BaseAppOutput):
    video: File = Field(description="Generated avatar video")


class App(BaseApp):
    async def setup(self, metadata):
        self.logger = logging.getLogger(__name__)
        self.client = get_client()
        self.cancel_flag = False
        self.logger.info("PixVerse Avatar initialized")

    async def on_cancel(self):
        self.cancel_flag = True
        return True

    async def run(self, input_data: AppInput) -> AppOutput:
        self.cancel_flag = False

        if not input_data.audio and not input_data.tts_text:
            raise ValueError("Provide either audio file or tts_text for avatar")

        img_id = await upload_image(self.client, input_data.image.path)

        input_metas = []
        try:
            from PIL import Image
            with Image.open(input_data.image.path) as im:
                input_metas.append(ImageMeta(width=im.width, height=im.height))
        except Exception:
            input_metas.append(ImageMeta())

        payload = {
            "img_id": img_id,
            "quality": input_data.quality,
        }

        if input_data.audio:
            self.logger.info("Audio avatar mode")
            audio_media_id = await upload_audio(self.client, input_data.audio.path)
            payload["audio_media_id"] = audio_media_id
        else:
            self.logger.info("TTS avatar mode")
            payload["lip_sync_tts_content"] = input_data.tts_text
            payload["lip_sync_tts_speaker_id"] = input_data.tts_speaker_id or "auto"

        if input_data.prompt:
            payload["prompt"] = input_data.prompt

        self.logger.info(f"Creating avatar: quality={input_data.quality}")
        resp = await api_post(self.client, "/openapi/v2/video/avatar/generate", payload)

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
            resolution=input_data.quality,
            extra={"model": "avatar"},
        )]
        if credits_used:
            outputs.append(RawMeta(cost=credits_used))

        return AppOutput(
            video=File(path=video_path),
            output_meta=OutputMeta(
                inputs=input_metas,
                outputs=outputs,
            ),
        )

    async def unload(self):
        await self.client.aclose()

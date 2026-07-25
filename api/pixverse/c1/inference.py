import logging
from typing import Optional

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta, ImageMeta
from pydantic import Field

from .pixverse_helper import get_client, upload_image, poll_video, download_file, api_post


class AppInput(BaseAppInput):
    prompt: str = Field(
        description="Text prompt describing the video (25-200 words recommended)",
    )
    image: Optional[File] = Field(
        default=None,
        description="Input image for image-to-video mode. PNG/JPG/JPEG/WebP, max 10000px. When provided, generates video from this image.",
    )
    quality: str = Field(
        default="720p",
        description="Output resolution",
        json_schema_extra={"enum": ["360p", "540p", "720p", "1080p"]},
    )
    duration: int = Field(
        default=5,
        ge=5,
        le=15,
        description="Video duration in seconds (5-15). Billed per second.",
    )
    aspect_ratio: str = Field(
        default="16:9",
        description="Video aspect ratio (text-to-video only; image-to-video uses image aspect ratio)",
        json_schema_extra={"enum": ["16:9", "9:16", "1:1", "4:3", "3:4"]},
    )
    negative_prompt: Optional[str] = Field(
        default=None,
        description="What to avoid in the video",
    )
    seed: Optional[int] = Field(
        default=None,
        ge=0,
        le=2147483647,
        description="Seed for reproducibility",
    )
    motion_mode: str = Field(
        default="normal",
        description="Motion intensity",
        json_schema_extra={"enum": ["normal", "fast"]},
    )


class AppOutput(BaseAppOutput):
    video: File = Field(description="Generated video file")


class App(BaseApp):
    async def setup(self, metadata):
        self.logger = logging.getLogger(__name__)
        self.client = get_client()
        self.cancel_flag = False
        self.logger.info("PixVerse C1 initialized")

    async def on_cancel(self):
        self.cancel_flag = True
        return True

    async def run(self, input_data: AppInput) -> AppOutput:
        self.cancel_flag = False
        input_metas = []

        if input_data.image:
            self.logger.info("Image-to-video mode")
            img_id = await upload_image(self.client, input_data.image.path)

            try:
                from PIL import Image
                with Image.open(input_data.image.path) as im:
                    input_metas.append(ImageMeta(width=im.width, height=im.height))
            except Exception:
                input_metas.append(ImageMeta())

            payload = {
                "img_id": img_id,
                "model": "c1",
                "duration": input_data.duration,
                "quality": input_data.quality,
                "motion_mode": input_data.motion_mode,
            }
            if input_data.prompt:
                payload["prompt"] = input_data.prompt
            if input_data.negative_prompt:
                payload["negative_prompt"] = input_data.negative_prompt
            if input_data.seed is not None:
                payload["seed"] = input_data.seed

            self.logger.info(f"Creating image-to-video: quality={input_data.quality}, duration={input_data.duration}s")
            resp = await api_post(self.client, "/openapi/v2/video/img/generate", payload)
        else:
            self.logger.info(f"Text-to-video mode: quality={input_data.quality}, duration={input_data.duration}s")
            payload = {
                "prompt": input_data.prompt,
                "model": "c1",
                "duration": input_data.duration,
                "quality": input_data.quality,
                "aspect_ratio": input_data.aspect_ratio,
                "motion_mode": input_data.motion_mode,
            }
            if input_data.negative_prompt:
                payload["negative_prompt"] = input_data.negative_prompt
            if input_data.seed is not None:
                payload["seed"] = input_data.seed

            resp = await api_post(self.client, "/openapi/v2/video/text/generate", payload)

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
                inputs=input_metas,
                outputs=[VideoMeta(
                    width=width,
                    height=height,
                    resolution=input_data.quality,
                    seconds=float(input_data.duration),
                    extra={
                        "model": "c1",
                        "mode": "image-to-video" if input_data.image else "text-to-video",
                    },
                )],
            ),
        )

    async def unload(self):
        await self.client.aclose()

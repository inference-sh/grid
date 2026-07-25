import logging
from typing import List, Optional

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta, ImageMeta
from pydantic import Field

from .pixverse_helper import get_client, upload_image, poll_video, download_file, api_post


class ImageReference(BaseAppInput):
    image: File = Field(description="Reference image")
    ref_name: str = Field(description="Reference name used in prompt with @ prefix (e.g. 'dog' → use '@dog ' in prompt)")
    type: str = Field(
        default="subject",
        description="Role of the image",
        json_schema_extra={"enum": ["subject", "background"]},
    )


class AppInput(BaseAppInput):
    prompt: str = Field(
        description="Scene description. Prefix ref_names with @ and include a space after (e.g. '@dog plays in the park').",
    )
    image_references: List[ImageReference] = Field(
        description="Reference images for composition (1-3). Each must have a unique ref_name.",
        min_length=1,
        max_length=3,
    )
    model: str = Field(
        default="v6",
        description="Generation model",
        json_schema_extra={"enum": ["c1", "v6"]},
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
        description="Video duration in seconds",
    )
    aspect_ratio: str = Field(
        default="16:9",
        description="Video aspect ratio",
        json_schema_extra={"enum": ["16:9", "9:16", "1:1", "4:3", "3:4"]},
    )
    seed: Optional[int] = Field(
        default=None,
        ge=0,
        le=2147483647,
        description="Seed for reproducibility",
    )


class AppOutput(BaseAppOutput):
    video: File = Field(description="Generated video file")


class App(BaseApp):
    async def setup(self, metadata):
        self.logger = logging.getLogger(__name__)
        self.client = get_client()
        self.cancel_flag = False
        self.logger.info("PixVerse Fusion initialized")

    async def on_cancel(self):
        self.cancel_flag = True
        return True

    async def run(self, input_data: AppInput) -> AppOutput:
        self.cancel_flag = False
        input_metas = []

        refs = []
        for ref in input_data.image_references:
            img_id = await upload_image(self.client, ref.image.path)
            refs.append({
                "img_id": img_id,
                "ref_name": ref.ref_name,
                "type": ref.type,
            })
            try:
                from PIL import Image
                with Image.open(ref.image.path) as im:
                    input_metas.append(ImageMeta(width=im.width, height=im.height))
            except Exception:
                input_metas.append(ImageMeta())

        self.logger.info(f"Uploaded {len(refs)} reference images")

        payload = {
            "image_references": refs,
            "prompt": input_data.prompt,
            "model": input_data.model,
            "duration": input_data.duration,
            "quality": input_data.quality,
            "aspect_ratio": input_data.aspect_ratio,
        }
        if input_data.seed is not None:
            payload["seed"] = input_data.seed

        self.logger.info(f"Creating fusion: model={input_data.model}, quality={input_data.quality}, duration={input_data.duration}s")
        resp = await api_post(self.client, "/openapi/v2/video/fusion/generate", payload)

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
                    extra={"model": input_data.model, "mode": "fusion"},
                )],
            ),
        )

    async def unload(self):
        await self.client.aclose()

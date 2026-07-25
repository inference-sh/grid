import logging
from typing import List, Optional

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta, RawMeta
from pydantic import Field

from .pixverse_helper import get_client, upload_video, upload_image, poll_video, download_file, api_post


class AppInput(BaseAppInput):
    video: File = Field(
        description="Source video to modify. MP4/MOV, max 100MB, max 1920px, max 30s.",
    )
    prompt: str = Field(
        description="Modification instruction. Use @selection0/@selection1 for mask refs and @img0/@img1 for image refs.",
    )
    reference_images: Optional[List[File]] = Field(
        default=None,
        description="Reference images for swap/add/restyle operations (max 3). Referenced as @img0, @img1, @img2 in prompt.",
    )
    quality: str = Field(
        default="540p",
        description="Output resolution (currently only 540p supported)",
        json_schema_extra={"enum": ["540p"]},
    )


class AppOutput(BaseAppOutput):
    video: File = Field(description="Modified video file")


class App(BaseApp):
    async def setup(self, metadata):
        self.logger = logging.getLogger(__name__)
        self.client = get_client()
        self.cancel_flag = False
        self.logger.info("PixVerse Modify initialized")

    async def on_cancel(self):
        self.cancel_flag = True
        return True

    async def run(self, input_data: AppInput) -> AppOutput:
        self.cancel_flag = False

        video_media_id = await upload_video(self.client, input_data.video.path)

        payload = {
            "video_media_id": video_media_id,
            "prompt": input_data.prompt,
            "quality": input_data.quality,
        }

        if input_data.reference_images:
            img_ids = []
            for img_file in input_data.reference_images:
                img_id = await upload_image(self.client, img_file.path)
                img_ids.append(img_id)
            payload["img_ids"] = img_ids
            self.logger.info(f"Uploaded {len(img_ids)} reference images")

        self.logger.info(f"Modifying video: quality={input_data.quality}")
        resp = await api_post(self.client, "/openapi/v2/video/modify/generate", payload)

        video_id = resp["video_id"]
        credits_used = resp.get("credit", 0)
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
            extra={"model": "modify"},
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

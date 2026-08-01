import os
import logging
from typing import Optional
from enum import Enum

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta, ImageMeta
from pydantic import Field

from .runway_helper import RunwayClient, download_file


class RatioEnum(str, Enum):
    r1280_720 = "1280:720"
    r720_1280 = "720:1280"
    r1104_832 = "1104:832"
    r832_1104 = "832:1104"
    r960_960 = "960:960"
    r1584_672 = "1584:672"


RATIO_DIMS = {
    "1280:720": (1280, 720),
    "720:1280": (720, 1280),
    "1104:832": (1104, 832),
    "832:1104": (832, 1104),
    "960:960": (960, 960),
    "1584:672": (1584, 672),
}


class AppInput(BaseAppInput):
    prompt: str = Field(
        description="Text prompt describing the video. Max 1000 chars.",
    )
    image: Optional[File] = Field(
        default=None,
        description="Input image for image-to-video. When provided, generates video starting from this image.",
    )
    ratio: RatioEnum = Field(
        default=RatioEnum.r1280_720,
        description="Output video aspect ratio.",
    )
    duration: int = Field(
        default=5,
        ge=2,
        le=10,
        description="Video duration in seconds (2-10).",
    )
    seed: Optional[int] = Field(
        default=None,
        description="Seed for reproducible results.",
    )


class AppOutput(BaseAppOutput):
    video: File = Field(description="Generated video file.")


class App(BaseApp):
    async def setup(self, metadata):
        self.logger = logging.getLogger(__name__)
        api_key = os.environ.get("RUNWAY_KEY")
        if not api_key:
            raise RuntimeError("RUNWAY_KEY must be set")
        self.client = RunwayClient(api_key=api_key, logger=self.logger)
        self.logger.info("Runway Gen-4.5 initialized")

    async def on_cancel(self):
        return True

    async def run(self, input_data: AppInput) -> AppOutput:
        is_i2v = input_data.image is not None
        mode = "image-to-video" if is_i2v else "text-to-video"
        self.logger.info(f"Mode: {mode}, ratio: {input_data.ratio.value}, duration: {input_data.duration}s")

        payload = {
            "model": "gen4.5",
            "promptText": input_data.prompt,
            "ratio": input_data.ratio.value,
            "duration": input_data.duration,
        }
        if input_data.seed is not None:
            payload["seed"] = input_data.seed

        input_metas = []
        if is_i2v:
            payload["promptImage"] = input_data.image.uri
            input_metas.append(ImageMeta())
            endpoint = "/v1/image_to_video"
        else:
            endpoint = "/v1/text_to_video"

        task = await self.client.create_task(endpoint, payload)
        self.logger.info(f"Task created: {task.id}")

        result = await self.client.poll_task(task.id)
        if not result.output:
            raise RuntimeError("No output in completed task")

        video_url = result.output[0]
        self.logger.info(f"Video ready: {video_url[:80]}...")
        video_path = await download_file(video_url, suffix=".mp4", logger=self.logger)

        w, h = RATIO_DIMS.get(input_data.ratio.value, (1280, 720))

        return AppOutput(
            video=File(path=video_path),
            output_meta=OutputMeta(
                inputs=input_metas,
                outputs=[VideoMeta(
                    width=w, height=h,
                    seconds=float(input_data.duration),
                    extra={"model": "gen4.5", "mode": mode},
                )],
            ),
        )

    async def unload(self):
        await self.client.close()

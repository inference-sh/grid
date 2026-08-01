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
    image: File = Field(
        description="Input image to animate into video.",
    )
    ratio: RatioEnum = Field(
        default=RatioEnum.r1280_720,
        description="Output video aspect ratio.",
    )
    duration: Optional[int] = Field(
        default=5,
        ge=2,
        le=10,
        description="Video duration in seconds.",
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
        self.logger.info("Runway Gen-4 Turbo initialized")

    async def on_cancel(self):
        return True

    async def run(self, input_data: AppInput) -> AppOutput:
        self.logger.info(f"Image-to-video, ratio: {input_data.ratio.value}, duration: {input_data.duration}s")

        payload = {
            "model": "gen4_turbo",
            "promptImage": input_data.image.uri,
            "promptText": input_data.prompt,
            "ratio": input_data.ratio.value,
        }
        if input_data.duration is not None:
            payload["duration"] = input_data.duration
        if input_data.seed is not None:
            payload["seed"] = input_data.seed

        task = await self.client.create_task("/v1/image_to_video", payload)
        self.logger.info(f"Task created: {task.id}")

        result = await self.client.poll_task(task.id)
        if not result.output:
            raise RuntimeError("No output in completed task")

        video_url = result.output[0]
        self.logger.info(f"Video ready: {video_url[:80]}...")
        video_path = await download_file(video_url, suffix=".mp4", logger=self.logger)

        w, h = RATIO_DIMS.get(input_data.ratio.value, (1280, 720))
        duration = float(input_data.duration) if input_data.duration else 5.0

        return AppOutput(
            video=File(path=video_path),
            output_meta=OutputMeta(
                inputs=[ImageMeta()],
                outputs=[VideoMeta(
                    width=w, height=h,
                    seconds=duration,
                    extra={"model": "gen4_turbo"},
                )],
            ),
        )

    async def unload(self):
        await self.client.close()

import os
import logging
from typing import List, Optional
from enum import Enum

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta
from pydantic import Field

from .runway_helper import RunwayClient, download_file


class AspectRatioEnum(str, Enum):
    r16_9 = "16:9"
    r4_3 = "4:3"
    r3_2 = "3:2"
    r1_1 = "1:1"
    r2_3 = "2:3"
    r3_4 = "3:4"
    r9_16 = "9:16"
    r21_9 = "21:9"


class Keyframe(BaseAppInput):
    seconds: float = Field(description="Timestamp in the input video for this keyframe.")
    image: File = Field(description="Reference image for this keyframe.")
    range: Optional[float] = Field(
        default=None,
        description="How many seconds around the keyframe timestamp to apply the guidance.",
    )


class AppInput(BaseAppInput):
    video: File = Field(
        description="Input video to transform. Max 16MB.",
    )
    prompt: Optional[str] = Field(
        default=None,
        description="Text prompt to guide the transformation.",
    )
    target_aspect_ratio: Optional[AspectRatioEnum] = Field(
        default=None,
        description="Target aspect ratio for the output video.",
    )
    keyframes: Optional[List[Keyframe]] = Field(
        default=None,
        description="Up to 5 keyframes with reference images at specific timestamps.",
    )
    seed: Optional[int] = Field(
        default=None,
        description="Seed for reproducible results.",
    )


class AppOutput(BaseAppOutput):
    video: File = Field(description="Transformed video file.")


class App(BaseApp):
    async def setup(self, metadata):
        self.logger = logging.getLogger(__name__)
        api_key = os.environ.get("RUNWAY_KEY")
        if not api_key:
            raise RuntimeError("RUNWAY_KEY must be set")
        self.client = RunwayClient(api_key=api_key, logger=self.logger)
        self.logger.info("Runway Aleph 2.0 initialized")

    async def on_cancel(self):
        return True

    async def run(self, input_data: AppInput) -> AppOutput:
        self.logger.info(f"Video-to-video with Aleph 2.0")

        payload = {
            "model": "aleph2",
            "videoUri": input_data.video.uri,
        }
        if input_data.prompt:
            payload["promptText"] = input_data.prompt
        if input_data.target_aspect_ratio:
            payload["targetAspectRatio"] = input_data.target_aspect_ratio.value
        if input_data.seed is not None:
            payload["seed"] = input_data.seed
        if input_data.keyframes:
            payload["keyframes"] = [
                {
                    "seconds": kf.seconds,
                    "uri": kf.image.uri,
                    **({"range": kf.range} if kf.range is not None else {}),
                }
                for kf in input_data.keyframes[:5]
            ]

        task = await self.client.create_task("/v1/video_to_video", payload)
        self.logger.info(f"Task created: {task.id}")

        result = await self.client.poll_task(task.id)
        if not result.output:
            raise RuntimeError("No output in completed task")

        video_url = result.output[0]
        self.logger.info(f"Video ready: {video_url[:80]}...")
        video_path = await download_file(video_url, suffix=".mp4", logger=self.logger)

        video_meta = VideoMeta.from_file(
            video_path,
            extra={"model": "aleph2"},
        )

        return AppOutput(
            video=File(path=video_path),
            output_meta=OutputMeta(
                inputs=[VideoMeta()],
                outputs=[video_meta],
            ),
        )

    async def unload(self):
        await self.client.close()

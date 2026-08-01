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
    r960_960 = "960:960"
    r1104_832 = "1104:832"
    r832_1104 = "832:1104"
    r1584_672 = "1584:672"


RATIO_DIMS = {
    "1280:720": (1280, 720),
    "720:1280": (720, 1280),
    "960:960": (960, 960),
    "1104:832": (1104, 832),
    "832:1104": (832, 1104),
    "1584:672": (1584, 672),
}


class CharacterTypeEnum(str, Enum):
    image = "image"
    video = "video"


class AppInput(BaseAppInput):
    character: File = Field(
        description="Character source — image (max 5MB) or video (max 16MB) of the character to animate.",
    )
    character_type: CharacterTypeEnum = Field(
        default=CharacterTypeEnum.image,
        description="Whether the character source is an image or video.",
    )
    reference_video: File = Field(
        description="Performance reference video (3-30 seconds, max 16MB). The character will mimic this performance.",
    )
    ratio: RatioEnum = Field(
        default=RatioEnum.r1280_720,
        description="Output video aspect ratio.",
    )
    body_control: bool = Field(
        default=False,
        description="Enable non-facial movements and gestures.",
    )
    expression_intensity: Optional[int] = Field(
        default=None,
        ge=1,
        le=5,
        description="Expression intensity (1-5). Higher = more expressive.",
    )
    seed: Optional[int] = Field(
        default=None,
        description="Seed for reproducible results.",
    )


class AppOutput(BaseAppOutput):
    video: File = Field(description="Generated character performance video.")


class App(BaseApp):
    async def setup(self, metadata):
        self.logger = logging.getLogger(__name__)
        api_key = os.environ.get("RUNWAY_KEY")
        if not api_key:
            raise RuntimeError("RUNWAY_KEY must be set")
        self.client = RunwayClient(api_key=api_key, logger=self.logger)
        self.logger.info("Runway Act-Two initialized")

    async def on_cancel(self):
        return True

    async def run(self, input_data: AppInput) -> AppOutput:
        self.logger.info(f"Character performance, type: {input_data.character_type.value}, ratio: {input_data.ratio.value}")

        payload = {
            "model": "act_two",
            "character": {
                "type": input_data.character_type.value,
                "uri": input_data.character.uri,
            },
            "reference": {
                "type": "video",
                "uri": input_data.reference_video.uri,
            },
            "ratio": input_data.ratio.value,
        }
        if input_data.body_control:
            payload["bodyControl"] = True
        if input_data.expression_intensity is not None:
            payload["expressionIntensity"] = input_data.expression_intensity
        if input_data.seed is not None:
            payload["seed"] = input_data.seed

        input_metas = [VideoMeta()]
        if input_data.character_type == CharacterTypeEnum.image:
            input_metas.append(ImageMeta())
        else:
            input_metas.append(VideoMeta())

        task = await self.client.create_task("/v1/character_performance", payload)
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
                    extra={"model": "act_two", "body_control": input_data.body_control},
                )],
            ),
        )

    async def unload(self):
        await self.client.close()

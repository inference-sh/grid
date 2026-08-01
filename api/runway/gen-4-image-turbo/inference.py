import os
import logging
from typing import List, Optional
from enum import Enum

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field

from .runway_helper import RunwayClient, download_file


class RatioEnum(str, Enum):
    r1280_720 = "1280:720"
    r720_1280 = "720:1280"
    r1104_832 = "1104:832"
    r832_1104 = "832:1104"
    r960_960 = "960:960"
    r1584_672 = "1584:672"
    r1920_1080 = "1920:1080"
    r1080_1920 = "1080:1920"


RATIO_DIMS = {
    "1280:720": (1280, 720),
    "720:1280": (720, 1280),
    "1104:832": (1104, 832),
    "832:1104": (832, 1104),
    "960:960": (960, 960),
    "1584:672": (1584, 672),
    "1920:1080": (1920, 1080),
    "1080:1920": (1080, 1920),
}


class ReferenceImage(BaseAppInput):
    image: File = Field(description="Reference image.")
    tag: Optional[str] = Field(
        default=None,
        description="Tag name to reference in prompt with @tagName syntax.",
    )


class AppInput(BaseAppInput):
    prompt: str = Field(
        description="Text prompt. Use @tagName to reference tagged images.",
    )
    image: File = Field(
        description="Reference image (required). Gen-4 Image Turbo transforms reference images guided by your prompt.",
    )
    ratio: RatioEnum = Field(
        default=RatioEnum.r1280_720,
        description="Output image aspect ratio.",
    )
    additional_references: Optional[List[ReferenceImage]] = Field(
        default=None,
        description="Additional reference images with optional tags.",
    )
    seed: Optional[int] = Field(
        default=None,
        description="Seed for reproducible results.",
    )


class AppOutput(BaseAppOutput):
    image: File = Field(description="Generated image.")


class App(BaseApp):
    async def setup(self, metadata):
        self.logger = logging.getLogger(__name__)
        api_key = os.environ.get("RUNWAY_KEY")
        if not api_key:
            raise RuntimeError("RUNWAY_KEY must be set")
        self.client = RunwayClient(api_key=api_key, logger=self.logger)
        self.logger.info("Runway Gen-4 Image Turbo initialized")

    async def on_cancel(self):
        return True

    async def run(self, input_data: AppInput) -> AppOutput:
        self.logger.info(f"Image-to-image (turbo), ratio: {input_data.ratio.value}")

        refs = [{"uri": input_data.image.uri}]
        input_metas = [ImageMeta()]

        if input_data.additional_references:
            for ref in input_data.additional_references:
                entry = {"uri": ref.image.uri}
                if ref.tag:
                    entry["tag"] = ref.tag
                refs.append(entry)
                input_metas.append(ImageMeta())

        payload = {
            "model": "gen4_image_turbo",
            "promptText": input_data.prompt,
            "ratio": input_data.ratio.value,
            "referenceImages": refs,
        }
        if input_data.seed is not None:
            payload["seed"] = input_data.seed

        task = await self.client.create_task("/v1/text_to_image", payload)
        self.logger.info(f"Task created: {task.id}")

        result = await self.client.poll_task(task.id)
        if not result.output:
            raise RuntimeError("No output in completed task")

        image_url = result.output[0]
        self.logger.info(f"Image ready: {image_url[:80]}...")
        image_path = await download_file(image_url, suffix=".png", logger=self.logger)

        w, h = RATIO_DIMS.get(input_data.ratio.value, (1280, 720))

        return AppOutput(
            image=File(path=image_path),
            output_meta=OutputMeta(
                inputs=input_metas,
                outputs=[ImageMeta(width=w, height=h, count=1)],
            ),
        )

    async def unload(self):
        await self.client.close()

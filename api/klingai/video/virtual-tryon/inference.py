"""
Kling Virtual Try-On - AI Clothing Try-On

Generate try-on images by combining a person photo with a clothing image.
Supports single clothing items (upper, lower, dress) and upper+lower combos (v1.5).
"""

import os
import logging
from enum import Enum

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field

from .kling_helper import (
    KlingClient,
    KlingAPIError,
    poll_task,
)
from .download_helper import download_file


class ModelEnum(str, Enum):
    v1 = "kolors-virtual-try-on-v1"
    v1_5 = "kolors-virtual-try-on-v1-5"


class AppInput(BaseAppInput):
    """Kling Virtual Try-On.

    Provide a person image and a clothing image to generate a try-on result.
    V1.5 supports upper+lower combo images on white background.
    """

    human_image: File = Field(
        description="Person photo to try clothes on. Clear full/half body shot. Formats: jpg, jpeg, png. Max 10MB, min 300px.",
    )
    cloth_image: File = Field(
        description="Clothing image - product photo or white background. Formats: jpg, jpeg, png. Max 10MB, min 300px. V1.5 supports upper+lower combos merged into one image.",
    )
    model: ModelEnum = Field(
        default=ModelEnum.v1_5,
        description="Model version. V1.5 supports upper+lower clothing combos.",
    )


class AppOutput(BaseAppOutput):
    image: File = Field(description="The try-on result image.")


class App(BaseApp):

    async def setup(self, metadata):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        logging.getLogger("httpx").setLevel(logging.WARNING)

        access_key = os.environ.get("KLING_ACCESS_KEY")
        secret_key = os.environ.get("KLING_SECRET_KEY")
        if not access_key or not secret_key:
            raise RuntimeError("KLING_ACCESS_KEY and KLING_SECRET_KEY must be set")

        self.client = KlingClient(access_key=access_key, secret_key=secret_key)
        self.logger.info("Kling Virtual Try-On initialized")

    async def on_cancel(self):
        self.logger.info("Cancellation requested")
        return True

    async def run(self, input_data: AppInput) -> AppOutput:
        self.logger.info(f"Creating try-on task, model: {input_data.model.value}")

        task = await self.client.virtual_tryon.create(
            human_image=input_data.human_image.uri,
            cloth_image=input_data.cloth_image.uri,
            model_name=input_data.model.value,
        )

        self.logger.info(f"Task created: {task.task_id}")

        result = await poll_task(
            self.client.virtual_tryon.get,
            task.task_id,
            interval=2.0,
            timeout=300.0,
        )

        if not result.images or not result.images[0].url:
            raise RuntimeError(f"No image URL in result: {result.task_status_msg}")

        image_url = result.images[0].url
        self.logger.info(f"Image ready: {image_url[:80]}...")

        image_path = download_file(image_url, suffix=".png", logger=self.logger)

        output_meta = OutputMeta(
            outputs=[
                ImageMeta(
                    count=1,
                    extra={
                        "model": input_data.model.value,
                    },
                )
            ]
        )

        return AppOutput(image=File(path=image_path), output_meta=output_meta)

    async def unload(self):
        await self.client.close()

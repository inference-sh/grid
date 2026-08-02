import os
import logging
from typing import List, Optional
from enum import Enum

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field

from .krea_helper import KreaClient, download_file


class TrainingTypeEnum(str, Enum):
    style = "Style"
    object = "Object"
    character = "Character"


class AppInput(BaseAppInput):
    name: str = Field(description="Name for the trained style.")
    images: List[File] = Field(description="Training images (5-20 recommended).")
    type: TrainingTypeEnum = Field(default=TrainingTypeEnum.style, description="Training type: style, object, or character.")
    trigger_word: Optional[str] = Field(default=None, description="Trigger word to activate the style in prompts.")
    max_train_steps: Optional[int] = Field(default=None, ge=1, le=2000, description="Maximum training steps (1-2000).")


class AppOutput(BaseAppOutput):
    style_id: str = Field(description="Trained style ID for use in Krea 2 Medium Turbo generation.")
    name: str = Field(description="Name of the trained style.")


class App(BaseApp):
    async def setup(self):
        self.logger = logging.getLogger(__name__)
        api_key = os.environ.get("KREA_KEY")
        if not api_key:
            raise RuntimeError("KREA_KEY must be set")
        self.client = KreaClient(api_key=api_key, logger=self.logger)
        self.logger.info("Krea 2 Medium Turbo LoRA trainer initialized")

    async def on_cancel(self):
        return True

    async def run(self, input_data: AppInput) -> AppOutput:
        self.logger.info(f"Training LoRA '{input_data.name}' with {len(input_data.images)} images")

        image_urls = []
        for i, img in enumerate(input_data.images):
            local_path = await download_file(img.uri, suffix=".png", logger=self.logger)
            asset_url = await self.client.upload_asset(local_path)
            image_urls.append(asset_url)
            self.logger.info(f"Uploaded image {i+1}/{len(input_data.images)}")

        payload = {
            "model": "k2",
            "name": input_data.name,
            "urls": image_urls,
            "type": input_data.type.value,
        }
        if input_data.trigger_word:
            payload["trigger_word"] = input_data.trigger_word
        if input_data.max_train_steps is not None:
            payload["max_train_steps"] = input_data.max_train_steps

        result = await self.client.train(payload)

        result_data = result.get("result") or {}
        style_id = result_data.get("id") or result_data.get("style_id") or result.get("id", "")

        self.logger.info(f"Training completed, style_id={style_id}")

        return AppOutput(
            style_id=str(style_id),
            name=input_data.name,
            output_meta=OutputMeta(
                inputs=[ImageMeta() for _ in input_data.images],
                outputs=[],
            ),
        )

    async def unload(self):
        await self.client.close()

"""
Kling Image V1.5 (Kolors V1.5) - Subject & Face Reference

Text-to-image with subject/face reference for character consistency.
Provide a reference image to generate new images matching the person's features.
"""

import os
import logging
from typing import Optional
from enum import Enum

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field

from .kling_helper import KlingClient, poll_task
from .download_helper import download_file


class AspectRatioEnum(str, Enum):
    r16_9 = "16:9"
    r9_16 = "9:16"
    r1_1 = "1:1"
    r4_3 = "4:3"
    r3_4 = "3:4"
    r3_2 = "3:2"
    r2_3 = "2:3"
    r21_9 = "21:9"


class ReferenceTypeEnum(str, Enum):
    subject = "subject"
    face = "face"


class AppInput(BaseAppInput):
    """Kolors V1.5 - text-to-image with subject/face reference.

    - subject: preserves character features (body, clothing style)
    - face: preserves facial appearance only (single face required)
    """

    prompt: str = Field(description="Text description. Max 2500 chars.")
    image: Optional[File] = Field(default=None, description="Reference image for subject/face consistency.")
    image_reference: Optional[ReferenceTypeEnum] = Field(default=None, description="Reference type: 'subject' for character features, 'face' for facial appearance.")
    image_fidelity: float = Field(default=0.5, ge=0.0, le=1.0, description="Reference strength [0-1].")
    human_fidelity: float = Field(default=0.45, ge=0.0, le=1.0, description="Facial similarity [0-1]. Only for 'subject' type.")
    negative_prompt: Optional[str] = Field(default=None, description="What to avoid. Not supported with image reference.")
    aspect_ratio: AspectRatioEnum = Field(default=AspectRatioEnum.r16_9, description="Output aspect ratio.")
    n: int = Field(default=1, ge=1, le=9, description="Number of images (1-9).")


class AppOutput(BaseAppOutput):
    image: File = Field(description="Generated image.")


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
        self.logger.info("Kling Image V1.5 initialized")

    async def on_cancel(self):
        return True

    async def run(self, input_data: AppInput) -> AppOutput:
        mode = "face-reference" if input_data.image_reference == ReferenceTypeEnum.face else "subject-reference" if input_data.image_reference else "text-to-image"
        self.logger.info(f"Mode: {mode}, n: {input_data.n}, ratio: {input_data.aspect_ratio.value}")

        task = await self.client.images.create(
            prompt=input_data.prompt,
            model_name="kling-v1-5",
            negative_prompt=input_data.negative_prompt if not input_data.image_reference else None,
            image=input_data.image.uri if input_data.image else None,
            image_reference=input_data.image_reference.value if input_data.image_reference else None,
            image_fidelity=input_data.image_fidelity,
            human_fidelity=input_data.human_fidelity,
            n=input_data.n,
            aspect_ratio=input_data.aspect_ratio.value,
        )
        self.logger.info(f"Task created: {task.task_id}")

        result = await poll_task(self.client.images.get, task.task_id, interval=2.0, timeout=300.0)
        if not result.images or not result.images[0].url:
            raise RuntimeError(f"No image URL: {result.task_status_msg}")

        image_path = download_file(result.images[0].url, suffix=".png", logger=self.logger)
        output_meta = OutputMeta(outputs=[ImageMeta(count=input_data.n, extra={"model": "kling-v1-5", "mode": mode})])
        return AppOutput(image=File(path=image_path), output_meta=output_meta)

    async def unload(self):
        await self.client.close()

"""
Kling Image Generation - Text-to-Image & Image-to-Image

Generate images using Kling's Kolors models. Supports text-to-image,
image-to-image with subject/face reference, and multi-image generation.
"""

import os
import logging
from typing import Optional
from enum import Enum

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field

from .kling_helper import (
    KlingClient,
    KlingAPIError,
    poll_task,
)
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


class ResolutionEnum(str, Enum):
    k1 = "1k"
    k2 = "2k"


class ReferenceTypeEnum(str, Enum):
    subject = "subject"
    face = "face"


class AppInput(BaseAppInput):
    """Kling image generation.

    Modes:
    - Text-to-image: prompt only
    - Image-to-image: prompt + image + image_reference (subject or face)
    """

    prompt: str = Field(
        description="Text description of the image to generate. Max 2500 chars.",
        examples=["A futuristic cityscape at night with neon lights and flying cars"],
    )
    image: Optional[File] = Field(
        default=None,
        description="Reference image for image-to-image generation. Requires image_reference to be set.",
    )
    image_reference: Optional[ReferenceTypeEnum] = Field(
        default=None,
        description="How to use the reference image: 'subject' for character features, 'face' for facial appearance.",
    )
    image_fidelity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Reference image strength [0-1]. Higher = closer to reference.",
    )
    human_fidelity: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="Facial similarity [0-1]. Only used with 'subject' reference type.",
    )
    negative_prompt: Optional[str] = Field(
        default=None,
        description="What to avoid in the image. Not supported with image reference.",
    )
    aspect_ratio: AspectRatioEnum = Field(
        default=AspectRatioEnum.r16_9,
        description="Output aspect ratio.",
    )
    resolution: ResolutionEnum = Field(
        default=ResolutionEnum.k1,
        description="Output resolution. 2k only supported by kling-v2.",
    )
    n: int = Field(
        default=1,
        ge=1,
        le=9,
        description="Number of images to generate (1-9). Each counts as one concurrency slot.",
    )


class AppOutput(BaseAppOutput):
    image: File = Field(description="The generated image file.")


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
        self.logger.info("Kling Image Generation initialized")

    async def on_cancel(self):
        self.logger.info("Cancellation requested")
        return True

    async def run(self, input_data: AppInput) -> AppOutput:
        model = "kling-v2-1"
        if input_data.image and input_data.image_reference:
            model = "kling-v1-5"
        if input_data.resolution == ResolutionEnum.k2:
            model = "kling-v2"

        mode = "image-to-image" if input_data.image else "text-to-image"
        self.logger.info(f"Mode: {mode}, model: {model}, n: {input_data.n}, ratio: {input_data.aspect_ratio.value}")
        self.logger.info(f"Prompt: {input_data.prompt[:100]}")

        task = await self.client.images.create(
            prompt=input_data.prompt,
            model_name=model,
            negative_prompt=input_data.negative_prompt if not input_data.image_reference else None,
            image=input_data.image.uri if input_data.image else None,
            image_reference=input_data.image_reference.value if input_data.image_reference else None,
            image_fidelity=input_data.image_fidelity,
            human_fidelity=input_data.human_fidelity,
            resolution=input_data.resolution.value,
            n=input_data.n,
            aspect_ratio=input_data.aspect_ratio.value,
        )

        self.logger.info(f"Task created: {task.task_id}")

        result = await poll_task(
            self.client.images.get,
            task.task_id,
            interval=2.0,
            timeout=300.0,
        )

        if not result.images or not result.images[0].url:
            raise RuntimeError(f"No image URL in result: {result.task_status_msg}")

        image_url = result.images[0].url
        self.logger.info(f"Image ready: {image_url[:80]}...")

        image_path = download_file(image_url, suffix=".png", logger=self.logger)

        # Estimate dimensions from aspect ratio
        base = 1024 if input_data.resolution == ResolutionEnum.k1 else 2048
        ratio_dims = {
            "16:9": (base, int(base * 9 / 16)),
            "9:16": (int(base * 9 / 16), base),
            "1:1": (base, base),
            "4:3": (base, int(base * 3 / 4)),
            "3:4": (int(base * 3 / 4), base),
            "3:2": (base, int(base * 2 / 3)),
            "2:3": (int(base * 2 / 3), base),
            "21:9": (base, int(base * 9 / 21)),
        }
        width, height = ratio_dims.get(input_data.aspect_ratio.value, (base, base))

        output_meta = OutputMeta(
            outputs=[
                ImageMeta(
                    width=width,
                    height=height,
                    count=input_data.n,
                    extra={
                        "mode": mode,
                        "model": model,
                        "resolution": input_data.resolution.value,
                    },
                )
            ]
        )

        return AppOutput(image=File(path=image_path), output_meta=output_meta)

    async def unload(self):
        await self.client.close()

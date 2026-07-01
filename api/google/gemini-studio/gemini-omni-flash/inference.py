"""
Gemini Omni Flash via Gemini API.

Text-to-video and image-to-video generation with synchronized audio,
grounded in Gemini's real-world knowledge.
Uses the Interactions API.
"""

import base64
import os
import tempfile
import logging

from enum import Enum
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta
from pydantic import Field
from typing import Optional, List

from google import genai


class VideoAspectRatioEnum(str, Enum):
    ratio_16_9 = "16:9"
    ratio_9_16 = "9:16"


class AppInput(BaseAppInput):
    prompt: str = Field(
        description="Text prompt describing the desired video content.",
        max_length=20000
    )
    image: Optional[File] = Field(
        None,
        description="Optional input image for image-to-video generation."
    )
    reference_images: Optional[List[File]] = Field(
        None,
        description="Optional reference images to guide video content (subject/style reference)."
    )
    aspect_ratio: VideoAspectRatioEnum = Field(
        default=VideoAspectRatioEnum.ratio_16_9,
        description="Video aspect ratio. 16:9 for landscape, 9:16 for portrait."
    )


class AppOutput(BaseAppOutput):
    video: File = Field(description="The generated video")


MIME_MAP = {
    '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
    '.png': 'image/png', '.webp': 'image/webp',
}


class App(BaseApp):
    async def setup(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.model_id = "gemini-omni-flash-preview"

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable is required.")
        self.client = genai.Client(api_key=api_key)
        self.logger.info("Gemini Omni Flash initialized")

    def _read_image_as_base64(self, file: File) -> tuple:
        ext = os.path.splitext(file.path)[1].lower()
        mime_type = MIME_MAP.get(ext, 'image/jpeg')
        with open(file.path, 'rb') as f:
            data = base64.b64encode(f.read()).decode('utf-8')
        return data, mime_type

    async def run(self, input_data: AppInput) -> AppOutput:
        try:
            has_image = input_data.image is not None
            has_refs = input_data.reference_images is not None and len(input_data.reference_images) > 0

            if has_image:
                if not input_data.image.exists():
                    raise RuntimeError(f"Input image does not exist: {input_data.image.path}")
                self.logger.info(f"Image-to-video: {input_data.prompt[:100]}...")
            elif has_refs:
                for i, ref in enumerate(input_data.reference_images):
                    if not ref.exists():
                        raise RuntimeError(f"Reference image {i+1} does not exist: {ref.path}")
                self.logger.info(f"Reference-to-video ({len(input_data.reference_images)} refs): {input_data.prompt[:100]}...")
            else:
                self.logger.info(f"Text-to-video: {input_data.prompt[:100]}...")

            # Build input
            if has_image:
                img_data, mime_type = self._read_image_as_base64(input_data.image)
                interaction_input = [
                    {"type": "image", "data": img_data, "mime_type": mime_type},
                    {"type": "text", "text": input_data.prompt},
                ]
                task = "image_to_video"
            elif has_refs:
                interaction_input = []
                for ref in input_data.reference_images:
                    ref_data, mime_type = self._read_image_as_base64(ref)
                    interaction_input.append({"type": "image", "data": ref_data, "mime_type": mime_type})
                interaction_input.append({"type": "text", "text": input_data.prompt})
                task = "reference_to_video"
            else:
                interaction_input = input_data.prompt
                task = "text_to_video"

            self.logger.info(f"Task: {task}, Aspect ratio: {input_data.aspect_ratio.value}")

            interaction = self.client.interactions.create(
                model=self.model_id,
                input=interaction_input,
                response_format={
                    "type": "video",
                    "aspect_ratio": input_data.aspect_ratio.value,
                },
                generation_config={
                    "video_config": {
                        "task": task,
                    }
                },
            )

            if not interaction.output_video or not interaction.output_video.data:
                raise RuntimeError("No video was generated")

            video_bytes = base64.b64decode(interaction.output_video.data)
            self.logger.info(f"Generated video: {len(video_bytes)} bytes")

            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(video_bytes)
                video_path = tmp.name

            self.logger.info(f"Video saved to {video_path}")

            if input_data.aspect_ratio == VideoAspectRatioEnum.ratio_16_9:
                width, height = 1280, 720
            else:
                width, height = 720, 1280

            return AppOutput(
                video=File(path=video_path),
                output_meta=OutputMeta(
                    outputs=[VideoMeta(width=width, height=height, seconds=8, resolution="720p")]
                )
            )

        except Exception as e:
            self.logger.error(f"Error: {e}")
            raise RuntimeError(f"Video generation failed: {str(e)}")

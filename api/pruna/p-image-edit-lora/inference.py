"""
P-Image-Edit-LoRA - Image editing with custom LoRA weights from HuggingFace
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field
from typing import Optional, List
from enum import Enum
import logging

from .pruna_helper import run_prediction, download_image, upload_file


class LoraPresetEnum(str, Enum):
    """Pre-trained edit LoRA styles available on HuggingFace."""
    dotted_illustration = "dotted-illustration"
    photo_enhancement = "photo-enhancement"
    skin_retouching = "skin-retouching"
    photo_to_anime = "photo-to-anime"
    next_scene = "next-scene"
    photo_upscaler = "photo-upscaler"
    enhance = "enhance"
    user_generated_content = "user-generated-content"


# Map preset enum to full HuggingFace URLs
LORA_PRESET_URLS = {
    LoraPresetEnum.dotted_illustration: "huggingface.co/davidberenstein1957/p-image-edit-dotted-illustration-lora/weights.safetensors",
    LoraPresetEnum.photo_enhancement: "huggingface.co/davidberenstein1957/p-image-edit-photo-enhancement-lora/weights.safetensors",
    LoraPresetEnum.skin_retouching: "huggingface.co/davidberenstein1957/p-image-edit-skin-retouching-lora/weights.safetensors",
    LoraPresetEnum.photo_to_anime: "huggingface.co/davidberenstein1957/p-image-edit-photo-to-anime-lora/weights.safetensors",
    LoraPresetEnum.next_scene: "huggingface.co/davidberenstein1957/p-image-edit-next-scene-lora/weights.safetensors",
    LoraPresetEnum.photo_upscaler: "huggingface.co/davidberenstein1957/p-image-edit-photo-upscaler-lora/weights.safetensors",
    LoraPresetEnum.enhance: "huggingface.co/davidberenstein1957/enhance/weights.safetensors",
    LoraPresetEnum.user_generated_content: "huggingface.co/davidberenstein1957/user_generated_content/weights.safetensors",
}


class AspectRatioEnum(str, Enum):
    match_input = "match_input_image"
    square = "1:1"
    landscape = "16:9"
    portrait = "9:16"
    photo_landscape = "4:3"
    photo_portrait = "3:4"
    classic_landscape = "3:2"
    classic_portrait = "2:3"


class AppInput(BaseAppInput):
    """Input schema for P-Image-Edit-LoRA."""

    prompt: str = Field(
        description="Text instruction describing the edit. Refer to images as 'image 1', 'image 2', etc."
    )
    images: List[File] = Field(
        description="Reference images for editing (main image first).",
        min_length=1,
        max_length=5,
    )
    lora_preset: LoraPresetEnum = Field(
        default=LoraPresetEnum.photo_enhancement,
        description="Pre-trained edit LoRA style."
    )
    lora_url: Optional[str] = Field(
        default=None,
        description="Custom LoRA URL (overrides preset). Format: huggingface.co/owner/repo/file.safetensors"
    )
    lora_scale: float = Field(
        default=1.0,
        ge=-1,
        le=3,
        description="LoRA strength (-1 to 3)."
    )
    hf_api_token: Optional[str] = Field(
        default=None,
        description="HuggingFace API token for private LoRAs."
    )
    turbo: bool = Field(
        default=True,
        description="Fast mode. Set false for complex tasks requiring more precision."
    )
    aspect_ratio: AspectRatioEnum = Field(
        default=AspectRatioEnum.match_input,
        description="Output aspect ratio."
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducible generation."
    )
    disable_safety_checker: bool = Field(
        default=False,
        description="Disable safety checker for generated images."
    )


class AppOutput(BaseAppOutput):
    """Output schema for P-Image-Edit-LoRA."""

    image: File = Field(description="Edited image file.")
    seed: Optional[int] = Field(default=None, description="Seed used for generation.")


class App(BaseApp):
    """P-Image-Edit-LoRA for image editing with custom LoRA."""

    async def setup(self, metadata):
        """Initialize the application."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.model = "p-image-edit-lora"
        self.logger.info("P-Image-Edit-LoRA initialized")

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Edit images using P-Image-Edit-LoRA."""
        try:
            # Resolve LoRA URL (custom url overrides preset)
            if input_data.lora_url:
                lora_url = input_data.lora_url
            else:
                lora_url = LORA_PRESET_URLS[input_data.lora_preset]

            self.logger.info(f"Editing with LoRA: {lora_url}")
            self.logger.info(f"Prompt: {input_data.prompt[:100]}...")

            # Upload images and collect URLs
            image_urls = []
            for i, img in enumerate(input_data.images):
                if not img.exists():
                    raise RuntimeError(f"Image {i+1} does not exist: {img.path}")

                if img.uri and img.uri.startswith("http"):
                    image_urls.append(img.uri)
                else:
                    self.logger.info(f"Uploading image {i+1}...")
                    upload_result = upload_file(img.path, logger=self.logger)
                    file_url = upload_result.get("urls", {}).get("get")
                    if not file_url:
                        raise RuntimeError(f"Failed to get URL for uploaded image {i+1}")
                    image_urls.append(file_url)

            # Build request
            request_data = {
                "prompt": input_data.prompt,
                "images": image_urls,
                "lora_weights": lora_url,
                "lora_scale": input_data.lora_scale,
                "turbo": input_data.turbo,
                "aspect_ratio": input_data.aspect_ratio.value,
            }

            if input_data.hf_api_token:
                request_data["hf_api_token"] = input_data.hf_api_token
            if input_data.seed is not None:
                request_data["seed"] = input_data.seed
            if input_data.disable_safety_checker:
                request_data["disable_safety_checker"] = input_data.disable_safety_checker

            # Run prediction
            result = await run_prediction(
                model=self.model,
                input_data=request_data,
                use_sync=True,
                logger=self.logger,
            )

            # Download result
            generation_url = result.get("generation_url")
            if not generation_url:
                raise RuntimeError("No generation_url in response")

            if generation_url.startswith("/"):
                generation_url = f"https://api.pruna.ai{generation_url}"

            image_path = download_image(generation_url, logger=self.logger)

            output_meta = OutputMeta(
                outputs=[ImageMeta(width=1024, height=1024, count=1)]
            )

            self.logger.info("Image edited successfully")

            return AppOutput(
                image=File(path=image_path),
                seed=result.get("seed"),
                output_meta=output_meta,
            )

        except Exception as e:
            self.logger.error(f"Error: {e}")
            raise RuntimeError(f"Image editing failed: {str(e)}")

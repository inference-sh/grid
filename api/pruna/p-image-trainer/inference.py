"""
P-Image-Trainer - Train custom LoRA weights for p-image-lora
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, RawMeta
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum
import logging
import zipfile
import tempfile
import os

from .pruna_helper import create_prediction, poll_prediction_status, download_result, upload_file


class TrainingTypeEnum(str, Enum):
    content = "content"
    style = "style"
    balanced = "balanced"


class TrainingImage(BaseModel):
    """An image with an optional caption for LoRA training."""
    image: File = Field(description="Training image file.")
    caption: Optional[str] = Field(
        default=None,
        description="Caption describing this image. Falls back to default_caption if not set."
    )


class AppInput(BaseAppInput):
    """Input schema for P-Image-Trainer."""

    images: Optional[List[File]] = Field(
        default=None,
        description="Drag-drop mode: array of training images (at least 10). All images use default_caption."
    )
    training_images: Optional[List[TrainingImage]] = Field(
        default=None,
        description="Captioned mode: array of {image, caption} pairs for per-image captions."
    )
    training_data: Optional[File] = Field(
        default=None,
        description="Pre-made ZIP with images and .txt caption files (e.g., photo.jpg + photo.txt). Power-user escape hatch."
    )
    steps: int = Field(
        default=1000,
        ge=100,
        le=5000,
        description="Number of training steps (100-5000, increments of 100). More steps = longer training, potentially better results."
    )
    learning_rate: float = Field(
        default=0.0001,
        ge=0.00001,
        le=0.01,
        description="Learning rate for training. Lower = slower but more stable."
    )
    training_type: TrainingTypeEnum = Field(
        default=TrainingTypeEnum.balanced,
        description="Type of training: content (subjects/characters), style (artistic styles), balanced (both)."
    )
    default_caption: Optional[str] = Field(
        default=None,
        description="Fallback caption for images without an explicit caption. Required if no per-image captions are provided."
    )


class AppOutput(BaseAppOutput):
    """Output schema for P-Image-Trainer."""

    lora_weights: File = Field(description="ZIP file containing trained LoRA weights (.safetensors). Download within 30 minutes.")


class App(BaseApp):
    """P-Image-Trainer LoRA training."""

    async def setup(self):
        """Initialize the application."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.model = "p-image-trainer"
        self.logger.info("P-Image-Trainer initialized")

    def _get_ext(self, path: str) -> str:
        """Get file extension, default to .jpg."""
        ext = os.path.splitext(path)[1].lower()
        return ext if ext else ".jpg"

    def _build_training_zip(self, images: List[File], captions: List[Optional[str]], default_caption: Optional[str]) -> str:
        """Build a training ZIP with sequentially named files and captions."""
        self.logger.info(f"Building training ZIP from {len(images)} images...")
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            zip_path = tmp.name

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for i, img in enumerate(images):
                ext = self._get_ext(img.path)
                name = f"image_{i + 1:04d}"

                # add image
                zf.write(img.path, f"{name}{ext}")

                # add caption
                caption = captions[i] if i < len(captions) and captions[i] is not None else default_caption
                if caption is not None:
                    zf.writestr(f"{name}.txt", caption)

        self.logger.info(f"Training ZIP ready ({os.path.getsize(zip_path)} bytes)")
        return zip_path

    def _upload_zip(self, zip_path: str) -> str:
        """Upload a ZIP file to Pruna and return the URL."""
        self.logger.info("Uploading training ZIP...")
        result = upload_file(zip_path, logger=self.logger)
        os.unlink(zip_path)

        file_url = result.get("url") or result.get("download_url") or (result.get("urls", {}).get("get"))
        if not file_url:
            raise RuntimeError(f"No URL in upload response: {result}")
        return file_url

    async def run(self, input_data: AppInput) -> AppOutput:
        """Train a LoRA using P-Image-Trainer."""
        try:
            has_images = input_data.images and len(input_data.images) > 0
            has_training_images = input_data.training_images and len(input_data.training_images) > 0
            has_training_data = input_data.training_data is not None

            if not has_images and not has_training_images and not has_training_data:
                raise ValueError("Provide 'images', 'training_images', or 'training_data'")

            self.logger.info(f"Starting LoRA training: {input_data.steps} steps, type={input_data.training_type.value}")

            # Resolve training data to a Pruna-accessible URL
            if has_training_images:
                # Structured pairs: extract files and captions
                files = [ti.image for ti in input_data.training_images]
                captions = [ti.caption for ti in input_data.training_images]
                self.logger.info(f"Captioned mode: {len(files)} image+caption pairs")
                zip_path = self._build_training_zip(files, captions, input_data.default_caption)
                image_data_url = self._upload_zip(zip_path)

            elif has_images:
                # Simple drag-drop: all images use default_caption
                self.logger.info(f"Drag-drop mode: {len(input_data.images)} images")
                zip_path = self._build_training_zip(input_data.images, [], input_data.default_caption)
                image_data_url = self._upload_zip(zip_path)

            else:
                # Pre-made ZIP
                self.logger.info("ZIP mode: uploading pre-made training data")
                image_data_url = self._upload_zip_file(input_data.training_data)

            # Build request
            request_data = {
                "image_data": image_data_url,
                "steps": input_data.steps,
                "training_type": input_data.training_type.value,
                "learning_rate": input_data.learning_rate,
            }

            if input_data.default_caption is not None:
                request_data["default_caption"] = input_data.default_caption

            # Submit async prediction (training is never sync)
            result = create_prediction(
                model=self.model,
                input_data=request_data,
                try_sync=False,
                logger=self.logger,
            )

            prediction_id = result.get("id")
            if not prediction_id:
                raise RuntimeError("No prediction ID in response")

            self.logger.info(f"Training submitted: {prediction_id}")

            # Poll for completion (training can take minutes to hours)
            completed = await poll_prediction_status(
                prediction_id=prediction_id,
                logger=self.logger,
                max_wait=7200.0,
            )

            # Download output ZIP
            generation_url = completed.get("generation_url") or completed.get("output")
            if not generation_url:
                raise RuntimeError("No output URL in completed prediction")
            if isinstance(generation_url, list):
                generation_url = generation_url[0]
            if generation_url.startswith("/"):
                generation_url = f"https://api.pruna.ai{generation_url}"

            self.logger.info("Training complete, downloading weights...")
            zip_path = download_result(generation_url, suffix=".zip", logger=self.logger)

            # Pricing: $1.80 per 1000 steps = 180 cents per 1000 steps
            cost_cents = (input_data.steps / 1000.0) * 180.0
            output_meta = OutputMeta(
                outputs=[RawMeta(cost=cost_cents)],
            )

            self.logger.info("LoRA training completed successfully")

            return AppOutput(
                lora_weights=File(path=zip_path),
                output_meta=output_meta,
            )

        except Exception as e:
            self.logger.error(f"Error: {e}")
            raise RuntimeError(f"LoRA training failed: {str(e)}")

    def _upload_zip_file(self, training_data: File) -> str:
        """Upload a pre-made ZIP file to Pruna."""
        self.logger.info("Uploading pre-made training ZIP...")
        result = upload_file(training_data.path, logger=self.logger)
        file_url = result.get("url") or result.get("download_url") or (result.get("urls", {}).get("get"))
        if not file_url:
            raise RuntimeError(f"No URL in upload response: {result}")
        return file_url

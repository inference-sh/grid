"""
P-Image-Edit-Trainer - Train custom LoRA weights for p-image-edit-lora from before/after pairs
"""

import logging
import os
import tempfile
import zipfile
from typing import List, Optional

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, RawMeta
from pydantic import BaseModel, Field, model_validator

from .pruna_helper import create_prediction, poll_prediction_status, download_result, file_url


class TrainingPair(BaseModel):
    """A before/after image pair with an optional edit instruction."""
    start: File = Field(description="Before image.")
    end: File = Field(description="After image (the transformed result).")
    extra_starts: List[File] = Field(
        default_factory=list,
        description="Additional before references for this pair (saved as ROOT_start2, ROOT_start3, ...).",
    )
    caption: Optional[str] = Field(default=None, description="Edit instruction for this pair. Falls back to default_caption.")


class AppInput(BaseAppInput):
    pairs: Optional[List[TrainingPair]] = Field(
        default=None,
        json_schema_extra={"x-promoted": True},
        description="Before/after image pairs. The app builds the training ZIP.",
    )
    training_data: Optional[File] = Field(
        default=None,
        description="Pre-made ZIP with ROOT_start.EXT / ROOT_end.EXT pairs and optional ROOT.txt captions.",
    )
    steps: int = Field(default=1000, ge=100, le=5000, multiple_of=100, description="Training steps (100-5000, multiples of 100).")
    learning_rate: float = Field(default=0.0001, ge=0.00001, le=0.01, description="Learning rate. Lower is slower and more stable.")
    default_caption: Optional[str] = Field(
        default=None,
        description="Edit instruction for pairs without a caption. Training fails when captions are missing and this is empty.",
    )

    @model_validator(mode="after")
    def check_data(self):
        if not self.pairs and not self.training_data:
            raise ValueError("Provide pairs or training_data")
        if self.pairs and not self.default_caption and any(not p.caption for p in self.pairs):
            raise ValueError("Every pair needs a caption, or set default_caption")
        return self


class AppOutput(BaseAppOutput):
    lora_weights: File = Field(description="ZIP with the trained LoRA weights (.safetensors) for p-image-edit-lora.")


def _ext(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return ext if ext else ".jpg"


class App(BaseApp):
    async def setup(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.model = "p-image-edit-trainer"

    def _build_zip(self, pairs: List[TrainingPair], default_caption: Optional[str]) -> str:
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            zip_path = tmp.name
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for i, pair in enumerate(pairs):
                root = f"pair_{i + 1:04d}"
                zf.write(pair.start.path, f"{root}_start{_ext(pair.start.path)}")
                for j, extra in enumerate(pair.extra_starts):
                    zf.write(extra.path, f"{root}_start{j + 2}{_ext(extra.path)}")
                zf.write(pair.end.path, f"{root}_end{_ext(pair.end.path)}")
                caption = pair.caption or default_caption
                if caption:
                    zf.writestr(f"{root}.txt", caption)
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
        self.logger.info("Training ZIP: %d pairs, %d bytes, files %s", len(pairs), os.path.getsize(zip_path), names[:12])
        return zip_path

    async def run(self, input_data: AppInput) -> AppOutput:
        if input_data.pairs:
            zip_path = self._build_zip(input_data.pairs, input_data.default_caption)
            image_data = file_url(File(path=zip_path), logger=self.logger)
            os.unlink(zip_path)
        else:
            image_data = file_url(input_data.training_data, logger=self.logger)

        request_data = {
            "image_data": image_data,
            "steps": input_data.steps,
            "learning_rate": input_data.learning_rate,
        }
        if input_data.default_caption:
            request_data["default_caption"] = input_data.default_caption

        self.logger.info("p-image-edit-trainer: %d steps, lr=%s", input_data.steps, input_data.learning_rate)

        # Training is async only
        result = create_prediction(model=self.model, input_data=request_data, try_sync=False, logger=self.logger)
        prediction_id = result.get("id")
        if not prediction_id:
            raise RuntimeError("No prediction ID in response")
        completed = await poll_prediction_status(prediction_id, logger=self.logger, max_wait=4 * 3600.0)

        self.logger.info("Prediction metadata: %s", completed.get("metadata"))
        url = completed.get("generation_url") or completed.get("output")
        if isinstance(url, list):
            url = url[0]
        if not isinstance(url, str):
            raise RuntimeError("No output URL in completed prediction")
        if url.startswith("/"):
            url = f"https://api.pruna.ai{url}"
        # The output URL expires ~30 minutes after training ends: download now
        zip_path = download_result(url, suffix=".zip", logger=self.logger)

        # $4.00 per 1000 steps, in cents
        cost_cents = (input_data.steps / 1000.0) * 400.0
        return AppOutput(
            lora_weights=File(path=zip_path),
            output_meta=OutputMeta(outputs=[RawMeta(cost=cost_cents)]),
        )

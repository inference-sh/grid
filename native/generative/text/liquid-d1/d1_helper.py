"""Shared by the liquid/d1-* apps (symlinked into each app directory).

Liquid AI's d1 models answer typed questions over a state in one forward pass and generate
no text. Both repositories ship their own model code and expose the same call:

    model.system_one(state, {id: question}, images=None)
    -> {"answers": {id: answer}, "usage": {"input_tokens": n, "output_tokens": 0}}

The input and output are the SDK's decision contract (inferencesh.models.decision). This
file holds what the two apps share besides: loading a pinned revision, opening images, and
writing a question's text positions as strings, which is all these models read.

Editing this file changes every app that links it. Redeploy all of them.
"""

import json
import time
from typing import Any, Dict, List

from inferencesh.models.decision import DecisionInput, Structured


def as_text(value: Structured) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def text_questions(input_data: DecisionInput) -> Dict[str, Dict[str, Any]]:
    """The input's questions with instructions, option descriptions and levels as strings."""
    questions = input_data.questions()
    for question in questions.values():
        question["instructions"] = as_text(question["instructions"])
        criteria = question.get("criteria")
        if isinstance(criteria, dict):
            question["criteria"] = {key: None if value is None else as_text(value) for key, value in criteria.items()}
        elif isinstance(criteria, list):
            question["criteria"] = [as_text(level) for level in criteria]
    return questions


# ── Model ────────────────────────────────────────────────────────────────────

def vram_gb(peak: bool = False) -> float:
    """GPU memory held by this process in GB (or its peak since the last reset); 0 on CPU."""
    import torch

    if not torch.cuda.is_available():
        return 0.0
    return (torch.cuda.max_memory_allocated() if peak else torch.cuda.memory_allocated()) / 1e9


def load_model(model_id: str, revision: str, dtype_name: str, logger) -> Any:
    """Download the pinned revision and load it with the repository's own code, in `dtype_name` on GPU."""
    import torch
    from accelerate import Accelerator
    from huggingface_hub import snapshot_download
    from transformers import AutoModel

    started = time.monotonic()
    logger.info(f"downloading {model_id}@{revision[:8]}")
    checkpoint = snapshot_download(model_id, revision=revision)
    logger.info(f"downloaded in {time.monotonic() - started:.1f}s")

    device = Accelerator().device
    dtype = torch.float32 if device.type == "cpu" else getattr(torch, dtype_name)
    started = time.monotonic()
    logger.info(f"loading on {device} as {dtype}")
    model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True, dtype=dtype).to(device).eval()
    logger.info(f"loaded in {time.monotonic() - started:.1f}s; {vram_gb():.1f} GB VRAM")
    return model


def open_images(files: List[Any]) -> List[Any]:
    """Input files as upright RGB PIL images."""
    from PIL import Image, ImageOps

    images = []
    for file in files:
        if not file.exists():
            raise RuntimeError(f"image does not exist at path: {file.path}")
        with Image.open(file.path) as image:
            images.append(ImageOps.exif_transpose(image).convert("RGB"))
    return images

"""
pplx-decider-v1.1-27b — Perplexity's open decision model, text and images, run on our own GPUs.

One request evaluates one `state` (text, plus up to 4 images) against any number of typed
questions and returns a probability for every option, in one forward pass per question.
No text is generated. Three question types:

  choice  which of these options?   -> choice, probabilities, confidence
  score   which level on a scale?   -> score, probabilities, confidence, legend
  noul    is this true?             -> noul (probability of yes, 0 to 1)

The input and output match `typesafe/jev` and the `decision-2-0-*` apps, with `images`
added, so a caller can move between them.

The checkpoint is a Qwen3.8-27B backbone with a separate 255-way readout, evaluated with
non-causal attention in its full-attention layers. It has no `lm_head`, so it does not
load in vLLM; it runs through the repository's own `autojev.model.DecisionModel`, which
also builds the prompt and applies the saved calibration temperature. That module needs
Python 3.12.

Metering: inputs=[TextMeta(tokens=<input tokens, summed over questions>)], outputs=[TextMeta(tokens=0)].

Model: https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b
"""

import asyncio
import logging
import sys
import time
from typing import Any, Dict, List

from inferencesh import BaseApp, File
from inferencesh.models.decision import DecisionOutput, DecisionVisionInput, Structured
from pydantic import Field

MODEL_ID = "perplexity-ai/pplx-decider-v1.1-27b"
# setup() imports source/src/autojev from the repository. Review the diff before moving this.
REVISION = "3b45dead91dfa6d95aad6b95764a606fab2bf7a6"
# Tokens per question: state, images, question and options. Over the limit is an error.
MAX_INPUT_TOKENS = 8192
# Questions per forward, as in the repository's own server.
BATCH_SIZE = 8
# Padded tokens in one forward. Bounds peak VRAM; a larger batch is split.
BATCH_TOKENS = 32768
MAX_IMAGES = 4


# ── Input and output ─────────────────────────────────────────────────────────

class AppInput(DecisionVisionInput):
    state: Structured = Field(
        default="",
        description="The text to evaluate: a string, or a JSON object / array of related context (messages, records, a policy). Every question sees the same state and images. May be empty when `images` carries the content. Input over the model's token limit is rejected, never truncated.",
        examples=["Seller title: wireless earbuds, barely used"],
    )
    images: List[File] = Field(
        default_factory=list,
        max_length=MAX_IMAGES,
        description="Up to 4 images the questions are about (PNG, JPEG or WebP). Every question sees them, placed before the state.",
    )


class AppOutput(DecisionOutput):
    model: str = Field(description="The model that answered: `perplexity-ai/pplx-decider-v1.1-27b`.")
    input_tokens: int = Field(default=0, description="Input tokens, summed over the questions. Image tokens are included.")


# ── Model ────────────────────────────────────────────────────────────────────

def vram_gb(peak: bool = False) -> float:
    """GPU memory held by this process in GB (or its peak since the last reset); 0 on CPU."""
    import torch

    if not torch.cuda.is_available():
        return 0.0
    return (torch.cuda.max_memory_allocated() if peak else torch.cuda.memory_allocated()) / 1e9


def load_model(logger) -> Any:
    """Download the pinned revision and load it through the repository's DecisionModel."""
    from accelerate import Accelerator
    from huggingface_hub import snapshot_download

    started = time.monotonic()
    logger.info(f"downloading {MODEL_ID}@{REVISION[:8]}")
    checkpoint = snapshot_download(MODEL_ID, revision=REVISION)
    logger.info(f"downloaded in {time.monotonic() - started:.1f}s")

    sys.path.insert(0, f"{checkpoint}/source/src")
    from autojev.model import DecisionModel

    device = str(Accelerator().device)
    started = time.monotonic()
    logger.info(f"loading on {device}")
    model = DecisionModel(checkpoint, device=device)
    logger.info(
        f"loaded in {time.monotonic() - started:.1f}s; attention {model.attention_mode}, "
        f"temperature {model.temperature:.4f}; {vram_gb():.1f} GB VRAM"
    )
    return model


def predict(model: Any, rows: List[Dict[str, Any]], ids: List[str]) -> tuple:
    """Probabilities per row and the input tokens used, in batches of at most BATCH_SIZE rows and
    BATCH_TOKENS padded tokens. Every row carries the same state and images, so the padded width of
    the first batch sizes the rest: a batch that is too wide is prepared once more, at most once."""
    probabilities: List[List[float]] = []
    input_tokens = 0
    per_batch = BATCH_SIZE
    start = 0
    while start < len(rows):
        chunk = rows[start:start + per_batch]
        try:
            batch = model.prepare(chunk, max_length=MAX_INPUT_TOKENS)
        except ValueError as error:
            # DecisionModel.prepare raises ValueError for an over-length question.
            raise RuntimeError(
                f"{error} This model's input limit is {MAX_INPUT_TOKENS} tokens per question "
                f"(state, images, question and options); questions: {ids[start:start + per_batch]}"
            ) from error
        fits = max(1, min(BATCH_SIZE, BATCH_TOKENS // batch.inputs["input_ids"].shape[1]))
        if len(chunk) > fits:
            per_batch = fits
            del batch
            continue
        values = (model(batch) / model.temperature).softmax(-1).cpu().tolist()
        probabilities += [row[:count] for row, count in zip(values, batch.counts)]
        input_tokens += batch.input_tokens
        start += len(chunk)
    return probabilities, input_tokens


def decide(model: Any, input_data: AppInput, logger) -> AppOutput:
    """Evaluate the state and images against every question. Blocking: call it in a thread."""
    import torch
    from autojev.model import answer, open_image

    for image in input_data.images:
        if not image.exists():
            raise RuntimeError(f"image does not exist at path: {image.path}")
    images = [open_image(image.path) for image in input_data.images]
    questions = input_data.questions()
    ids = list(questions)
    rows = [{"state": input_data.state, "question": questions[qid], "images": images} for qid in ids]

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    with torch.inference_mode():
        distributions, input_tokens = predict(model, rows, ids)
    answers = {qid: answer(questions[qid], values) for qid, values in zip(ids, distributions)}
    elapsed_ms = (time.monotonic() - started) * 1000

    logger.info(
        f"answered by {MODEL_ID}: {input_tokens} input tokens, {len(rows)} questions, "
        f"{len(images)} images, {elapsed_ms:.1f} ms, peak {vram_gb(peak=True):.1f} GB VRAM"
    )
    return AppOutput.from_answers(answers, input_data, model=MODEL_ID, input_tokens=input_tokens)


# ── App ──────────────────────────────────────────────────────────────────────

class App(BaseApp):
    async def setup(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        self.model = await asyncio.to_thread(load_model, self.logger)

    async def run(self, input_data: AppInput) -> AppOutput:
        """Evaluate the state and images against every question."""
        return await asyncio.to_thread(decide, self.model, input_data, self.logger)

    async def unload(self):
        import torch

        if hasattr(self, "model"):
            del self.model
        torch.cuda.empty_cache()

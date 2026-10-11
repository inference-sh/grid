"""
Decision 3.0 (d3) — the multimodal decision models of vLLM Semantic Router, run on our own GPUs.

One request evaluates one `state`, optionally with images and videos, against any number of
typed questions and returns a typed answer with a probability for every option. No text is
generated. Three question types:

  choice  which of these options?   -> choice, probabilities, confidence
  score   which level on a scale?   -> score, probabilities, confidence, legend
  noul    is this true?             -> noul (probability of yes, 0 to 1)

The input and output are the shared decision contract (`DecisionVideoInput`, `DecisionOutput`),
so a caller can move between d3, `typesafe/jev`, `autotrust/jev-27b-vl` and the Decision 2.0
apps by changing the app name.

The repositories ship their own runtime (`trust_remote_code`, d3_runtime.py): each question is
one forward pass over the whole prompt, with the full-attention layers reading it in both
directions, so questions share no prefix and run as padded batches. Images go before the text,
videos after the images. Shared by every d3 app: each app's inference.py names its model, the
pinned revision and its token budget.

Metering: inputs=[TextMeta(tokens=<input tokens, summed over questions>)], outputs=[TextMeta(tokens=0)].

Models: https://huggingface.co/collections/vllm-sr/decision-30-6aca5f47b0d942380f6e4a64
"""

import os
import time
from typing import Any, Dict, List

from inferencesh import File
from inferencesh.models.decision import DecisionOutput, DecisionVideoInput, Structured
from pydantic import Field

# Prompt tokens per question: state, images, videos, question and options. The runtime has no
# limit of its own; this one keeps a single question inside one card.
MAX_INPUT_TOKENS = 32768
MAX_IMAGES = 8
MAX_VIDEOS = 4
# The runtime's default, and the batch sizes its warm-up compiles kernels for.
MAX_BATCH = 8
MAX_IDS_IN_ERROR = 8

ERROR_HELP = {
    "max_length_exceeded": f"the state, images and videos plus this question exceed {MAX_INPUT_TOKENS} input tokens; "
    "input is never truncated, so shorten the state or send fewer or smaller images and videos",
    "invalid_question": "the model runtime rejected the question as malformed",
    "invalid_model_output": "the model returned a non-finite result for this question",
}


# ── Input and output ─────────────────────────────────────────────────────────

class AppInput(DecisionVideoInput):
    state: Structured = Field(
        default="",
        description="The text to evaluate: a string, or a JSON object / array of related context (messages, records, a policy). Every question sees the same state, images and videos. May be empty when `images` or `videos` carry the content. Input over the model's token limit is rejected, never truncated.",
        examples=["The order arrived damaged yesterday. The customer has a receipt and asks for a replacement today."],
    )
    images: List[File] = Field(
        default_factory=list,
        max_length=MAX_IMAGES,
        description="Up to 8 images (PNG, JPEG or WebP) the questions are about, placed before the state. Each is read at up to 1.6 megapixels; larger images cost more tokens and time.",
    )
    videos: List[File] = Field(
        default_factory=list,
        max_length=MAX_VIDEOS,
        description="Up to 4 videos (MP4, WebM, MOV or MKV, at most 32 MB and 5 minutes each) the questions are about, placed after the images. Read at 2 frames per second, at most 32 frames per video at up to 0.2 megapixels; all videos together take at most 16,384 tokens.",
    )


class AppOutput(DecisionOutput):
    model: str = Field(description="The model that answered, e.g. `d3-mini`.")
    input_tokens: int = Field(default=0, description="Input tokens, summed over the questions. Image and video tokens are included.")


# ── Model ────────────────────────────────────────────────────────────────────

def vram_gb(peak: bool = False) -> float:
    """GPU memory held by this process in GB (or its peak since the last reset); 0 on CPU."""
    import torch

    if not torch.cuda.is_available():
        return 0.0
    return (torch.cuda.max_memory_allocated() if peak else torch.cuda.memory_allocated()) / 1e9


def load_model(model_id: str, revision: str, logger) -> Any:
    """Load a d3 repository at a pinned revision through its own runtime, then compile its kernels.

    The Gated DeltaNet kernels specialize on batch size and length class, so a fresh process pays
    seconds on the first request of each new shape; the runtime's warm-up compiles them all here.
    """
    import torch
    from accelerate import Accelerator
    from transformers import AutoModel

    # The runtime calls torch.cuda.set_device, which needs an index; accelerate may give bare "cuda".
    device = Accelerator().device
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    device = str(device)
    started = time.monotonic()
    logger.info(f"loading {model_id}@{revision[:8]} on {device}")
    model = AutoModel.from_pretrained(
        model_id, revision=revision, trust_remote_code=True, device=device,
        batch_size=MAX_BATCH, token=os.environ.get("HF_TOKEN"),
    )
    runtime = model.runtime
    runtime.max_length = MAX_INPUT_TOKENS
    logger.info(
        f"loaded {runtime.model_name} in {time.monotonic() - started:.1f}s; {vram_gb():.1f} GB VRAM; "
        f"kernels {runtime.kernels}; fast path: {runtime.fast_skipped or 'on'}"
    )
    logger.info(f"warmed up in {runtime.warmup():.1f}s")
    return runtime


def batch_size(widest: int, batch_tokens: int) -> int:
    """Questions per forward: as many as fit `batch_tokens` padded tokens, 1 to MAX_BATCH."""
    return max(1, min(MAX_BATCH, batch_tokens // max(widest, 1)))


def widest_prompt(prepared: Any) -> int:
    """Tokens in the longest question prompt of a prepared request (0 when none can run)."""
    lengths = list(prepared.lengths.values()) + [len(s) for s in prepared.sequences.values()]
    return max(lengths, default=0)


def raise_failed(answers: Dict[str, Dict[str, Any]], asked: int) -> None:
    """Fail the request when any question has no answer, grouped by reason."""
    failed: Dict[str, List[str]] = {}
    for qid, answer in answers.items():
        if answer.get("error"):
            failed.setdefault(str(answer["error"]), []).append(qid)
    if not failed:
        return
    count = sum(len(ids) for ids in failed.values())
    parts = []
    for reason, ids in failed.items():
        shown = ", ".join(f"'{qid}'" for qid in ids[:MAX_IDS_IN_ERROR])
        if len(ids) > MAX_IDS_IN_ERROR:
            shown += f" and {len(ids) - MAX_IDS_IN_ERROR} more"
        parts.append(f"{reason} for {shown}: {ERROR_HELP.get(reason, 'unknown error')}")
    raise RuntimeError(f"{count} of {asked} questions could not be answered. " + "; ".join(parts) + ".")


def decide(runtime: Any, input_data: AppInput, batch_tokens: int, logger) -> AppOutput:
    """Evaluate the state, images and videos against every question. Blocking: call it in a thread.

    `batch_tokens` caps the padded tokens of one forward, so peak VRAM is bounded by the app, not
    by the request; answers do not depend on the split.
    """
    import torch

    questions = input_data.questions()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    try:
        prepared = runtime.prepare(
            input_data.state, questions,
            images=[image.path for image in input_data.images] or None,
            videos=[video.path for video in input_data.videos] or None,
        )
    except ValueError as exc:
        raise ValueError(f"the request could not be read: {exc}") from exc
    widest = widest_prompt(prepared)
    runtime.batch_size = batch_size(widest, batch_tokens)
    probabilities, tokens = runtime.run(prepared)
    data = runtime.respond(prepared, probabilities, tokens)
    elapsed_ms = (time.monotonic() - started) * 1000

    answers = data.get("answers") or {}
    raise_failed(answers, len(questions))
    logger.info(
        f"answered by {data['model']}: {tokens} input tokens, {len(questions)} questions, "
        f"{len(input_data.images)} images, {len(input_data.videos)} videos, widest {widest} tokens, "
        f"batch {runtime.batch_size}, {elapsed_ms:.1f} ms, peak {vram_gb(peak=True):.1f} GB VRAM"
    )
    return AppOutput.from_answers(answers, input_data, model=data["model"], input_tokens=tokens)

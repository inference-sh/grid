"""
Decision 2.0 — the open decision models of vLLM Semantic Router, run on our own GPUs.

One request evaluates one `state` against any number of typed questions and returns a
typed answer with a probability for every option, in a single forward pass. No text is
generated. Three question types:

  choice  which of these options?   -> choice, probabilities, confidence
  score   which level on a scale?   -> score, probabilities, confidence, legend
  noul    is this true?             -> noul (probability of yes, 0 to 1)

The model takes one `questions` map keyed by id, discriminated by `type`. Here each type
has its own typed list (`choices`, `scores`, `nouls`) and every question carries its `id`;
answers come back keyed by the same ids. Ids share one namespace. The input and output
match `typesafe/jev`, so a caller can move between the two.

`state`, `instructions` and every option / level / criteria description accept a string,
an object or an array.

Shared by every decision-2-0-* app: each app's inference.py names its model and the
pinned revision. The repositories ship their own runtime (`trust_remote_code`), so the
revision is pinned to a reviewed commit.

Metering: inputs=[TextMeta(tokens=<input tokens>)], outputs=[TextMeta(tokens=0)].

Models: https://huggingface.co/collections/vllm-sr/decision-20-6ab7cf7bdfb506bf8269cb00
"""

import time
from typing import Any, Dict, List

from inferencesh.models.decision import DecisionInput, DecisionOutput, Structured
from pydantic import Field

MAX_IDS_IN_ERROR = 8

ERROR_HELP = {
    "max_length_exceeded": "the state plus this question's instructions and options exceed the input limit; "
    "input is never truncated, so shorten the state or use a Decision 2.0 model with a longer context",
    "invalid_question": "the model runtime rejected the question as malformed",
    "invalid_model_output": "the model returned a non-finite result for this question",
}


# ── Input and output ─────────────────────────────────────────────────────────

class AppInput(DecisionInput):
    state: Structured = Field(
        description="The content to evaluate: a string, or a JSON object / array of related context (messages, records, a policy). Text only. Every question sees the same state. Input over the model's token limit is rejected, never truncated.",
        examples=["The order arrived damaged yesterday. The customer has a receipt and asks for a replacement today."],
    )


class AppOutput(DecisionOutput):
    model: str = Field(description="The model that answered, e.g. `Decision-2.0-Sol-2B`.")
    input_tokens: int = Field(default=0, description="Input tokens, summed over the questions.")


# ── Model ────────────────────────────────────────────────────────────────────

def vram_gb(peak: bool = False) -> float:
    """GPU memory held by this process in GB (or its peak since the last reset); 0 on CPU."""
    import torch

    if not torch.cuda.is_available():
        return 0.0
    return (torch.cuda.max_memory_allocated() if peak else torch.cuda.memory_allocated()) / 1e9


def load_model(model_id: str, revision: str, batch_tokens: int, logger) -> Any:
    """Load a Decision 2.0 repository at a pinned revision through its own runtime.

    The runtime answers every question of a request in one forward unless its backend has
    a token budget, and activation memory grows with questions x tokens: 64 questions over
    a 4k-token state took 17 GB on the 0.6B model. `batch_tokens` caps the padded tokens
    of one forward, so peak VRAM is bounded by the app, not by the request. The runtime
    splits a larger request into several forwards; answers do not depend on the split.
    """
    from accelerate import Accelerator
    from transformers import AutoModel

    device = str(Accelerator().device)
    started = time.monotonic()
    logger.info(f"loading {model_id}@{revision[:8]} on {device}")
    model = AutoModel.from_pretrained(model_id, revision=revision, trust_remote_code=True, device=device)

    if batch_tokens < model.max_input_tokens:
        raise ValueError(f"batch_tokens {batch_tokens} is below the model's input limit {model.max_input_tokens}")
    backend = model.runtime.backend
    if not hasattr(backend, "batch_tokens"):
        raise RuntimeError("this revision's runtime has no forward token budget; review decision2/qwen.py before moving REVISION")
    backend.batch_tokens = min(backend.batch_tokens or batch_tokens, batch_tokens)

    logger.info(
        f"loaded {model.model_name} in {time.monotonic() - started:.1f}s; "
        f"input limit {model.max_input_tokens} tokens; {backend.batch_tokens} tokens per forward; "
        f"{vram_gb():.1f} GB VRAM"
    )
    return model


def decide(model: Any, input_data: AppInput, logger) -> AppOutput:
    """Evaluate the state against every question. Blocking: call it in a thread."""
    import torch

    questions = input_data.questions()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    data = model.system_one(state=input_data.state, questions=questions)
    elapsed_ms = (time.monotonic() - started) * 1000

    answers = data.get("answers") or {}
    failed: Dict[str, List[str]] = {}
    for qid, answer in answers.items():
        if answer.get("error"):
            failed.setdefault(str(answer["error"]), []).append(qid)
    if failed:
        count = sum(len(ids) for ids in failed.values())
        parts = []
        for reason, ids in failed.items():
            shown = ", ".join(f"'{qid}'" for qid in ids[:MAX_IDS_IN_ERROR])
            if len(ids) > MAX_IDS_IN_ERROR:
                shown += f" and {len(ids) - MAX_IDS_IN_ERROR} more"
            parts.append(f"{reason} for {shown}: {ERROR_HELP.get(reason, 'unknown error')}")
        suffix = ""
        if "max_length_exceeded" in failed:
            suffix = f" This model's input limit is {getattr(model, 'max_input_tokens', None)} tokens per question."
        raise RuntimeError(f"{count} of {len(questions)} questions could not be answered. " + "; ".join(parts) + "." + suffix)

    usage = data.get("usage") or {}
    input_tokens = int(usage.get("input_tokens") or 0)
    model_name = data.get("model") or getattr(model, "model_name", "")
    logger.info(
        f"answered by {model_name}: {input_tokens} input tokens, "
        f"{len(questions)} questions, {elapsed_ms:.1f} ms, peak {vram_gb(peak=True):.1f} GB VRAM"
    )

    return AppOutput.from_answers(answers, input_data, model=model_name, input_tokens=input_tokens)

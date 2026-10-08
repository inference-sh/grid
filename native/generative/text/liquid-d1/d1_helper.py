"""Shared by the liquid/d1-* apps (symlinked into each app directory).

Liquid AI's d1 models answer typed questions over a state in one forward pass and generate
no text. Both repositories ship their own model code and expose the same call:

    model.system_one(state, {id: question}, images=None)
    -> {"answers": {id: answer}, "usage": {"input_tokens": n, "output_tokens": 0}}

This file holds what the apps have in common: the question and answer schemas (the same as
`typesafe/jev` and the other decision apps), the mapping to the models' question format, and
loading a pinned revision.

Editing this file changes every app that links it. Redeploy all of them.
"""

import time
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

# Plain text, or JSON structure the model reads by key.
Structured = Union[str, Dict[str, Any], List[Any]]


# ── Questions ────────────────────────────────────────────────────────────────

class ChoiceOption(BaseModel):
    name: str = Field(min_length=1, description="Option name. Returned as `choice` and used as the key in `probabilities`. Sent to the model.")
    description: Optional[str] = Field(
        default=None,
        description="What this option covers. Omit when the name is clear on its own.",
    )


class ChoiceQuestion(BaseModel):
    """Which of these options? For a fixed set of unordered options."""
    id: str = Field(min_length=1, description="Your key for this question; the answer comes back under it. Not sent to the model.")
    instructions: str = Field(min_length=1, description="What the model should decide, written as a complete question.")
    options: List[ChoiceOption] = Field(
        min_length=2,
        max_length=255,
        description="The answer options (2 to 255). Give the full list, and add an `other` option when the list might not cover every input.",
    )


class ScoreQuestion(BaseModel):
    """Which level? For a position on a spectrum you can describe."""
    id: str = Field(min_length=1, description="Your key for this question; the answer comes back under it. Not sent to the model.")
    instructions: str = Field(min_length=1, description="What the model should rate, written as a complete question.")
    levels: List[str] = Field(
        min_length=2,
        max_length=10,
        description="Ordered level descriptions, low end to high end (2 to 10). A level's number is its index, starting at 0.",
    )


class NoulCriteria(BaseModel):
    true: Optional[str] = Field(default=None, description="What a yes (value near 1) means.")
    false: Optional[str] = Field(default=None, description="What a no (value near 0) means.")


class NoulQuestion(BaseModel):
    """Is this true? For a clean yes/no where the probability itself is the signal."""
    id: str = Field(min_length=1, description="Your key for this question; the answer comes back under it. Not sent to the model.")
    instructions: str = Field(
        min_length=1,
        description="The yes/no question, or a statement to judge. Make the boundary between yes and no unambiguous.",
    )
    criteria: Optional[NoulCriteria] = Field(default=None, description="Optional. Pins down a subtle yes/no boundary.")


def check_questions(choices: List[ChoiceQuestion], scores: List[ScoreQuestion], nouls: List[NoulQuestion]) -> None:
    """Raise ValueError unless there is at least one question and every id and option name is unique."""
    ids = [q.id for q in (*choices, *scores, *nouls)]
    if not ids:
        raise ValueError("ask at least one question in `choices`, `scores` or `nouls`")
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ValueError(f"question ids must be unique across choices, scores and nouls; repeated: {dupes}")
    for q in choices:
        names = [o.name for o in q.options]
        if len(set(names)) != len(names):
            raise ValueError(f"choice '{q.id}' has repeated option names")


def build_questions(choices: List[ChoiceQuestion], scores: List[ScoreQuestion], nouls: List[NoulQuestion]) -> Dict[str, Dict[str, Any]]:
    """The models' question format (`type`, `instructions`, `criteria`), keyed by question id."""
    questions: Dict[str, Dict[str, Any]] = {}
    for q in choices:
        questions[q.id] = {
            "type": "choice",
            "instructions": q.instructions,
            "criteria": {o.name: o.description for o in q.options},
        }
    for q in scores:
        questions[q.id] = {"type": "score", "instructions": q.instructions, "criteria": list(q.levels)}
    for q in nouls:
        question: Dict[str, Any] = {"type": "noul", "instructions": q.instructions}
        criteria = q.criteria.model_dump(exclude_none=True) if q.criteria is not None else {}
        if criteria:
            question["criteria"] = criteria
        questions[q.id] = question
    return questions


# ── Answers ──────────────────────────────────────────────────────────────────

class ChoiceAnswer(BaseModel):
    choice: str = Field(description="The highest-probability option.")
    confidence: float = Field(description="The probability of `choice`, 0 to 1. Gate actions on it; thresholds scale with risk.")
    probabilities: Dict[str, float] = Field(description="Every option mapped to its probability. Sums to 1.")


class ScoreAnswer(BaseModel):
    score: float = Field(description="Probability-weighted position on the levels, 0 to the top level number. Can land between levels.")
    normalized: float = Field(description="`score` divided by the top level number: 0 to 1, comparable across scales of different length.")
    confidence: float = Field(description="The probability of the most likely level, 0 to 1.")
    probabilities: Dict[str, float] = Field(description="Each level number (as a string) mapped to its probability. Sums to 1.")
    legend: Dict[str, Any] = Field(description="Each level number mapped back to its description.")


class NoulAnswer(BaseModel):
    noul: float = Field(description="Probability the answer is yes. Near 1 strong yes, near 0 strong no, near 0.5 uncertain. Threshold it in code.")


def build_answers(answers: Dict[str, Dict[str, Any]], questions: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Sort the models' answers into `choices`, `scores` and `nouls`, as the app outputs hold them."""
    missing = sorted(set(questions) - set(answers))
    if missing:
        raise RuntimeError(f"the model returned no answer for: {missing}")
    choices: Dict[str, ChoiceAnswer] = {}
    scores: Dict[str, ScoreAnswer] = {}
    nouls: Dict[str, NoulAnswer] = {}
    for qid, question in questions.items():
        result = answers[qid]
        if question["type"] == "choice":
            choices[qid] = ChoiceAnswer(choice=result["choice"], confidence=result["confidence"], probabilities=result["probabilities"])
        elif question["type"] == "score":
            scores[qid] = ScoreAnswer(
                score=result["score"],
                normalized=result["score"] / (len(question["criteria"]) - 1),
                confidence=result["confidence"],
                probabilities=result["probabilities"],
                legend=result["legend"],
            )
        else:
            nouls[qid] = NoulAnswer(noul=result["noul"])
    return {"choices": choices, "scores": scores, "nouls": nouls}


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

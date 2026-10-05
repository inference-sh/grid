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
from typing import Any, Dict, List, Optional, Union

from inferencesh import BaseAppInput, BaseAppOutput, OutputMeta, TextMeta
from pydantic import BaseModel, Field, model_validator

# Plain text, or JSON structure the model reads by key.
Structured = Union[str, Dict[str, Any], List[Any]]

MAX_IDS_IN_ERROR = 8

ERROR_HELP = {
    "max_length_exceeded": "the state plus this question's instructions and options exceed the input limit; "
    "input is never truncated, so shorten the state or use a Decision 2.0 model with a longer context",
    "invalid_question": "the model runtime rejected the question as malformed",
    "invalid_model_output": "the model returned a non-finite result for this question",
}


# ── Questions ────────────────────────────────────────────────────────────────

class ChoiceOption(BaseModel):
    name: str = Field(min_length=1, description="Option name. Returned as `choice` and used as the key in `probabilities`. Sent to the model.")
    description: Optional[Structured] = Field(
        default=None,
        description="What this option covers. Omit when the name is clear on its own. An object can carry a rubric, e.g. {what, not_for, examples}.",
    )


class ChoiceQuestion(BaseModel):
    """Which of these options? For a fixed set of unordered options."""
    id: str = Field(min_length=1, description="Your key for this question; the answer comes back under it. Not sent to the model.")
    instructions: Structured = Field(
        description="What the model should decide, written as a complete question. An object can hold the question in one field and data it refers to in others.",
    )
    options: List[ChoiceOption] = Field(
        min_length=2,
        max_length=255,
        description="The answer options (2 to 255). Give the full list, and add an `other` option when the list might not cover every input.",
    )


class ScoreQuestion(BaseModel):
    """Which level? For a position on a spectrum you can describe."""
    id: str = Field(min_length=1, description="Your key for this question; the answer comes back under it. Not sent to the model.")
    instructions: Structured = Field(description="What the model should rate, written as a complete question.")
    levels: List[Structured] = Field(
        min_length=2,
        max_length=10,
        description="Ordered level descriptions, low end to high end (2 to 10). A level's number is its index, starting at 0.",
    )


class NoulCriteria(BaseModel):
    true: Optional[Structured] = Field(default=None, description="What a yes (value near 1) means. Defaults to `Yes`.")
    false: Optional[Structured] = Field(default=None, description="What a no (value near 0) means. Defaults to `No`.")


class NoulQuestion(BaseModel):
    """Is this true? For a clean yes/no where the probability itself is the signal."""
    id: str = Field(min_length=1, description="Your key for this question; the answer comes back under it. Not sent to the model.")
    instructions: Structured = Field(
        description="The yes/no question, or a statement to judge. Make the boundary between yes and no unambiguous.",
    )
    criteria: Optional[NoulCriteria] = Field(default=None, description="Optional. Pins down a subtle yes/no boundary.")


class AppInput(BaseAppInput):
    state: Structured = Field(
        description="The content to evaluate: a string, or a JSON object / array of related context (messages, records, a policy). Text only. Every question sees the same state. Input over the model's token limit is rejected, never truncated.",
        examples=["The order arrived damaged yesterday. The customer has a receipt and asks for a replacement today."],
    )
    choices: List[ChoiceQuestion] = Field(default_factory=list, description="Choice questions: pick one option from a set.")
    scores: List[ScoreQuestion] = Field(default_factory=list, description="Score questions: place the state on ordered levels.")
    nouls: List[NoulQuestion] = Field(default_factory=list, description="Noul questions: probability that the answer is yes.")

    @model_validator(mode="after")
    def _check_questions(self):
        ids = [q.id for q in (*self.choices, *self.scores, *self.nouls)]
        if not ids:
            raise ValueError("ask at least one question in `choices`, `scores` or `nouls`")
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"question ids must be unique across choices, scores and nouls; repeated: {dupes}")
        for q in (*self.choices, *self.scores, *self.nouls):
            if q.instructions == "":
                raise ValueError(f"question '{q.id}' has empty instructions")
        for q in self.choices:
            names = [o.name for o in q.options]
            if len(set(names)) != len(names):
                raise ValueError(f"choice '{q.id}' has repeated option names")
        return self


# ── Answers ──────────────────────────────────────────────────────────────────

class ChoiceAnswer(BaseModel):
    choice: str = Field(description="The highest-probability option.")
    confidence: float = Field(description="0 to 1, from how peaked `probabilities` is. Gate actions on it; thresholds scale with risk.")
    probabilities: Dict[str, float] = Field(description="Every option mapped to its probability. Sums to 1.")


class ScoreAnswer(BaseModel):
    score: float = Field(description="Probability-weighted position on the levels, 0 to the top level number. Can land between levels.")
    normalized: float = Field(description="`score` divided by the top level number: 0 to 1, comparable across scales of different length.")
    confidence: float = Field(description="0 to 1, from how peaked `probabilities` is.")
    probabilities: Dict[str, float] = Field(description="Each level number (as a string) mapped to its probability. Sums to 1.")
    legend: Dict[str, Any] = Field(description="Each level number mapped back to its description.")


class NoulAnswer(BaseModel):
    noul: float = Field(description="Probability the answer is yes. Near 1 strong yes, near 0 strong no, near 0.5 uncertain. Threshold it in code.")


class AppOutput(BaseAppOutput):
    choices: Dict[str, ChoiceAnswer] = Field(default_factory=dict, description="Choice answers by question id.")
    scores: Dict[str, ScoreAnswer] = Field(default_factory=dict, description="Score answers by question id.")
    nouls: Dict[str, NoulAnswer] = Field(default_factory=dict, description="Noul answers by question id.")
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


def build_questions(input_data: AppInput) -> Dict[str, Dict[str, Any]]:
    questions: Dict[str, Dict[str, Any]] = {}
    for q in input_data.choices:
        questions[q.id] = {
            "type": "choice",
            "instructions": q.instructions,
            "criteria": {o.name: o.description for o in q.options},
        }
    for q in input_data.scores:
        questions[q.id] = {"type": "score", "instructions": q.instructions, "criteria": q.levels}
    for q in input_data.nouls:
        question: Dict[str, Any] = {"type": "noul", "instructions": q.instructions}
        if q.criteria is not None:
            criteria = q.criteria.model_dump(exclude_none=True)
            if criteria:
                question["criteria"] = criteria
        questions[q.id] = question
    return questions


def decide(model: Any, input_data: AppInput, logger) -> AppOutput:
    """Evaluate the state against every question. Blocking: call it in a thread."""
    import torch

    questions = build_questions(input_data)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    data = model.system_one(state=input_data.state, questions=questions)
    elapsed_ms = (time.monotonic() - started) * 1000

    answers = data.get("answers") or {}
    missing = sorted(set(questions) - set(answers))
    if missing:
        raise RuntimeError(f"the model returned no answer for: {missing}")
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

    top_level = {q.id: len(q.levels) - 1 for q in input_data.scores}
    choices: Dict[str, ChoiceAnswer] = {}
    scores: Dict[str, ScoreAnswer] = {}
    nouls: Dict[str, NoulAnswer] = {}
    for qid, answer in answers.items():
        kind = answer.get("type")
        if kind == "choice":
            choices[qid] = ChoiceAnswer(
                choice=answer.get("choice") or "",
                confidence=answer.get("confidence", 0.0),
                probabilities=answer.get("probabilities") or {},
            )
        elif kind == "score":
            score = answer.get("score", 0.0)
            scores[qid] = ScoreAnswer(
                score=score,
                normalized=score / top_level[qid] if top_level.get(qid) else 0.0,
                confidence=answer.get("confidence", 0.0),
                probabilities=answer.get("probabilities") or {},
                legend=answer.get("legend") or {},
            )
        elif kind == "noul":
            nouls[qid] = NoulAnswer(noul=answer.get("noul", 0.0))
        else:
            raise RuntimeError(f"answer '{qid}' has unknown type {kind!r}; keys: {list(answer.keys())}")

    usage = data.get("usage") or {}
    input_tokens = int(usage.get("input_tokens") or 0)
    model_name = data.get("model") or getattr(model, "model_name", "")
    logger.info(
        f"answered by {model_name}: {input_tokens} input tokens, "
        f"{len(questions)} questions, {elapsed_ms:.1f} ms, peak {vram_gb(peak=True):.1f} GB VRAM"
    )

    return AppOutput(
        choices=choices,
        scores=scores,
        nouls=nouls,
        model=model_name,
        input_tokens=input_tokens,
        output_meta=OutputMeta(
            inputs=[TextMeta(tokens=input_tokens)],
            outputs=[TextMeta(tokens=0)],
        ),
    )

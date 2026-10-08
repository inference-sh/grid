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
from typing import Any, Dict, List, Optional, Union

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, TextMeta
from pydantic import BaseModel, Field, model_validator

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

# Plain text, or JSON structure the model reads by key.
Structured = Union[str, Dict[str, Any], List[Any]]


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
    true: Optional[Structured] = Field(default=None, description="What a yes (value near 1) means. Defaults to `Yes / true`.")
    false: Optional[Structured] = Field(default=None, description="What a no (value near 0) means. Defaults to `No / false`.")


class NoulQuestion(BaseModel):
    """Is this true? For a clean yes/no where the probability itself is the signal."""
    id: str = Field(min_length=1, description="Your key for this question; the answer comes back under it. Not sent to the model.")
    instructions: Structured = Field(
        description="The yes/no question, or a statement to judge. Make the boundary between yes and no unambiguous.",
    )
    criteria: Optional[NoulCriteria] = Field(default=None, description="Optional. Pins down a subtle yes/no boundary.")


class AppInput(BaseAppInput):
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
        if self.state == "" and not self.images:
            raise ValueError("send a `state`, `images`, or both")
        return self


# ── Answers ──────────────────────────────────────────────────────────────────

class ChoiceAnswer(BaseModel):
    choice: str = Field(description="The highest-probability option.")
    confidence: float = Field(description="0 to 1: how far the top probability is above an even split. Gate actions on it; thresholds scale with risk.")
    probabilities: Dict[str, float] = Field(description="Every option mapped to its probability. Sums to 1.")


class ScoreAnswer(BaseModel):
    score: float = Field(description="Probability-weighted position on the levels, 0 to the top level number. Can land between levels.")
    normalized: float = Field(description="`score` divided by the top level number: 0 to 1, comparable across scales of different length.")
    confidence: float = Field(description="0 to 1, from how tightly `probabilities` sits around its top level.")
    probabilities: Dict[str, float] = Field(description="Each level number (as a string) mapped to its probability. Sums to 1.")
    legend: Dict[str, Any] = Field(description="Each level number mapped back to its description.")


class NoulAnswer(BaseModel):
    noul: float = Field(description="Probability the answer is yes. Near 1 strong yes, near 0 strong no, near 0.5 uncertain. Threshold it in code.")


class AppOutput(BaseAppOutput):
    choices: Dict[str, ChoiceAnswer] = Field(default_factory=dict, description="Choice answers by question id.")
    scores: Dict[str, ScoreAnswer] = Field(default_factory=dict, description="Score answers by question id.")
    nouls: Dict[str, NoulAnswer] = Field(default_factory=dict, description="Noul answers by question id.")
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


def predict(model: Any, rows: List[Dict[str, Any]]) -> tuple:
    """Probabilities per row and the input tokens used. Halves a batch whose padded size is over BATCH_TOKENS."""
    batch = model.prepare(rows, max_length=MAX_INPUT_TOKENS)
    padded = batch.inputs["input_ids"].numel()
    if len(rows) > 1 and padded > BATCH_TOKENS:
        del batch
        half = len(rows) // 2
        left, left_tokens = predict(model, rows[:half])
        right, right_tokens = predict(model, rows[half:])
        return left + right, left_tokens + right_tokens
    probabilities = (model(batch) / model.temperature).softmax(-1).cpu().tolist()
    return [values[:count] for values, count in zip(probabilities, batch.counts)], batch.input_tokens


def decide(model: Any, input_data: AppInput, logger) -> AppOutput:
    """Evaluate the state and images against every question. Blocking: call it in a thread."""
    import torch
    from autojev.model import answer, open_image

    for image in input_data.images:
        if not image.exists():
            raise RuntimeError(f"image does not exist at path: {image.path}")
    images = [open_image(image.path) for image in input_data.images]
    questions = build_questions(input_data)
    ids = list(questions)
    rows = [{"state": input_data.state, "question": questions[qid], "images": images} for qid in ids]

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    answers: Dict[str, Dict[str, Any]] = {}
    input_tokens = 0
    with torch.inference_mode():
        for start in range(0, len(rows), BATCH_SIZE):
            try:
                distributions, tokens = predict(model, rows[start:start + BATCH_SIZE])
            except ValueError as error:
                # DecisionModel.prepare raises ValueError for an over-length question.
                raise RuntimeError(
                    f"{error} This model's input limit is {MAX_INPUT_TOKENS} tokens per question "
                    f"(state, images, question and options); questions: {ids[start:start + BATCH_SIZE]}"
                ) from error
            input_tokens += tokens
            for qid, values in zip(ids[start:start + BATCH_SIZE], distributions):
                answers[qid] = answer(questions[qid], values)
    elapsed_ms = (time.monotonic() - started) * 1000

    choices: Dict[str, ChoiceAnswer] = {}
    scores: Dict[str, ScoreAnswer] = {}
    nouls: Dict[str, NoulAnswer] = {}
    top_level = {q.id: len(q.levels) - 1 for q in input_data.scores}
    for qid, result in answers.items():
        kind = result["type"]
        if kind == "choice":
            choices[qid] = ChoiceAnswer(choice=result["choice"], confidence=result["confidence"], probabilities=result["probabilities"])
        elif kind == "score":
            scores[qid] = ScoreAnswer(
                score=result["score"],
                normalized=result["score"] / top_level[qid],
                confidence=result["confidence"],
                probabilities=result["probabilities"],
                legend=result["legend"],
            )
        else:
            nouls[qid] = NoulAnswer(noul=result["noul"])

    logger.info(
        f"answered by {MODEL_ID}: {input_tokens} input tokens, {len(rows)} questions, "
        f"{len(images)} images, {elapsed_ms:.1f} ms, peak {vram_gb(peak=True):.1f} GB VRAM"
    )
    return AppOutput(
        choices=choices,
        scores=scores,
        nouls=nouls,
        model=MODEL_ID,
        input_tokens=input_tokens,
        output_meta=OutputMeta(
            inputs=[TextMeta(tokens=input_tokens)],
            outputs=[TextMeta(tokens=0)],
        ),
    )


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

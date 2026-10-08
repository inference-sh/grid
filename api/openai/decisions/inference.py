"""
OpenAI Decisions — typed decisions over text and images through OpenAI's Decisions API.

One request sends one `state` (text or JSON, plus images) and any number of typed questions
to `POST /v1/decisions` and returns a probability for every answer. No text is generated.
The input and output match `typesafe/jev` and the other decision apps, with `images` added,
so a caller can move between them. OpenAI's names map as follows:

  choice  ->  choice     value / description per option
  score   ->  score      one level per entry, label = the level text
  noul    ->  predicate  `criteria` is appended to the instructions

Questions go upstream as q0, q1, ... and answers are mapped back to the caller's ids, so an
id can be any string. Images are sent inline as base64 data URLs; the API takes no hosted
URLs or file ids.

Metering: inputs=[TextMeta(tokens=<input tokens>)], outputs=[TextMeta(tokens=0)].
OpenAI bills input tokens only ($0.10 per 1M for gpt-6-luna).

Docs: https://developers.openai.com/api/docs/guides/decisions
"""

import asyncio
import base64
import json
import logging
import mimetypes
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, TextMeta
from pydantic import BaseModel, Field, model_validator

API_URL = "https://api.openai.com/v1/decisions"
MAX_IMAGES = 8
MAX_ATTEMPTS = 4
RETRY_STATUSES = {429, 500, 502, 503, 504}

# Plain text, or JSON structure sent to the model as JSON text.
Structured = Union[str, Dict[str, Any], List[Any]]


# ── Questions ────────────────────────────────────────────────────────────────

class ChoiceOption(BaseModel):
    name: str = Field(min_length=1, description="Option name. Returned as `choice` and used as the key in `probabilities`. Sent to the model.")
    description: Optional[str] = Field(default=None, description="When this option applies. Omit when the name is clear on its own.")


class ChoiceQuestion(BaseModel):
    """Which of these options? For a fixed set of unordered options."""
    id: str = Field(min_length=1, description="Your key for this question; the answer comes back under it. Not sent to the model.")
    instructions: str = Field(min_length=1, description="What the model should decide, written as a complete question.")
    options: List[ChoiceOption] = Field(
        min_length=2,
        description="The answer options (at least 2). Give the full list, and add an `other` option when the list might not cover every input.",
    )


class ScoreQuestion(BaseModel):
    """Which level? For a position on a spectrum you can describe."""
    id: str = Field(min_length=1, description="Your key for this question; the answer comes back under it. Not sent to the model.")
    instructions: str = Field(min_length=1, description="What the model should rate, written as a complete question.")
    levels: List[str] = Field(
        min_length=2,
        description="Ordered level descriptions, low end to high end. A level's number is its index, starting at 0.",
    )


class NoulCriteria(BaseModel):
    true: Optional[str] = Field(default=None, description="What a yes (value near 1) means.")
    false: Optional[str] = Field(default=None, description="What a no (value near 0) means.")


class NoulQuestion(BaseModel):
    """Is this true? For a clean yes/no where the probability itself is the signal. OpenAI calls this a predicate."""
    id: str = Field(min_length=1, description="Your key for this question; the answer comes back under it. Not sent to the model.")
    instructions: str = Field(
        min_length=1,
        description="The yes/no question, or a statement to judge. Make the boundary between yes and no unambiguous.",
    )
    criteria: Optional[NoulCriteria] = Field(default=None, description="Optional. Pins down a subtle yes/no boundary; appended to the instructions.")


class AppInput(BaseAppInput):
    state: Structured = Field(
        default="",
        description="The text to evaluate: a string, or a JSON object / array of related context, sent as JSON text. Every question sees the same state and images. May be empty when `images` carries the content.",
        examples=["Seller title: wireless earbuds, barely used"],
    )
    images: List[File] = Field(
        default_factory=list,
        max_length=MAX_IMAGES,
        description="Up to 8 images the questions are about (PNG, JPEG or WebP). Every question sees them, placed before the state.",
    )
    choices: List[ChoiceQuestion] = Field(default_factory=list, description="Choice questions: pick one option from a set.")
    scores: List[ScoreQuestion] = Field(default_factory=list, description="Score questions: place the state on ordered levels.")
    nouls: List[NoulQuestion] = Field(default_factory=list, description="Noul questions: probability that the answer is yes.")
    model: str = Field(default="gpt-6-luna", description="The OpenAI model that evaluates the questions. The Decisions API supports `gpt-6-luna` only for now.")

    @model_validator(mode="after")
    def _check(self):
        ids = [q.id for q in (*self.choices, *self.scores, *self.nouls)]
        if not ids:
            raise ValueError("ask at least one question in `choices`, `scores` or `nouls`")
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"question ids must be unique across choices, scores and nouls; repeated: {dupes}")
        for q in self.choices:
            names = [o.name for o in q.options]
            if len(set(names)) != len(names):
                raise ValueError(f"choice '{q.id}' has repeated option names")
        if self.state == "" and not self.images:
            raise ValueError("send a `state`, `images`, or both")
        return self


# ── Answers ──────────────────────────────────────────────────────────────────

class ChoiceAnswer(BaseModel):
    choice: str = Field(description="The selected option.")
    confidence: float = Field(description="OpenAI's confidence in the answer, 0 to 1. Gate actions on it; thresholds scale with risk.")
    probabilities: Dict[str, float] = Field(description="Every option mapped to its probability.")


class ScoreAnswer(BaseModel):
    score: float = Field(description="Probability-weighted position on the levels, 0 to the top level number. Can land between levels.")
    normalized: float = Field(description="`score` divided by the top level number: 0 to 1, comparable across scales of different length.")
    confidence: float = Field(description="OpenAI's confidence in the answer, 0 to 1.")
    probabilities: Dict[str, float] = Field(description="Each level number (as a string) mapped to its probability.")
    legend: Dict[str, Any] = Field(description="Each level number mapped back to its description.")


class NoulAnswer(BaseModel):
    noul: float = Field(description="Probability the answer is yes. Near 1 strong yes, near 0 strong no, near 0.5 uncertain. Threshold it in code.")


class AppOutput(BaseAppOutput):
    choices: Dict[str, ChoiceAnswer] = Field(default_factory=dict, description="Choice answers by question id.")
    scores: Dict[str, ScoreAnswer] = Field(default_factory=dict, description="Score answers by question id.")
    nouls: Dict[str, NoulAnswer] = Field(default_factory=dict, description="Noul answers by question id.")
    model: str = Field(description="The model that answered.")
    input_tokens: int = Field(default=0, description="Input tokens (billed). Image tokens are included.")


# ── Request and response mapping ─────────────────────────────────────────────

def image_data_url(file: File) -> str:
    """An input image as the inline base64 data URL the API requires."""
    if not file.exists():
        raise RuntimeError(f"image does not exist at path: {file.path}")
    mime = file.content_type or mimetypes.guess_type(file.path)[0] or "image/png"
    with open(file.path, "rb") as handle:
        return f"data:{mime};base64,{base64.b64encode(handle.read()).decode()}"


def build_input(input_data: AppInput) -> Union[str, List[Dict[str, Any]]]:
    """The API's `input`: the state as a string, or one user message with the images then the state."""
    text = input_data.state if isinstance(input_data.state, str) else json.dumps(input_data.state, ensure_ascii=False)
    if not input_data.images:
        return text
    content: List[Dict[str, Any]] = [{"type": "input_image", "image_url": image_data_url(image)} for image in input_data.images]
    if text:
        content.append({"type": "input_text", "text": text})
    return [{"role": "user", "content": content}]


def predicate_instructions(question: NoulQuestion) -> str:
    """A noul's instructions with its criteria appended, since a predicate has no criteria field."""
    criteria = question.criteria.model_dump(exclude_none=True) if question.criteria is not None else {}
    if not criteria:
        return question.instructions
    return f"{question.instructions} ({'; '.join(f'{side}: {text}' for side, text in criteria.items())})"


def build_questions(input_data: AppInput) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """The API's `questions`, named q0, q1, ..., and the map from those names back to the caller's ids."""
    questions: List[Dict[str, Any]] = []
    for q in input_data.choices:
        options = [{"value": o.name, **({"description": o.description} if o.description else {})} for o in q.options]
        questions.append({"type": "choice", "instructions": q.instructions, "choices": options})
    for q in input_data.scores:
        questions.append({"type": "score", "instructions": q.instructions, "levels": [{"label": level} for level in q.levels]})
    for q in input_data.nouls:
        questions.append({"type": "predicate", "instructions": predicate_instructions(q)})
    ids = [q.id for q in (*input_data.choices, *input_data.scores, *input_data.nouls)]
    names: Dict[str, str] = {}
    for index, (question, qid) in enumerate(zip(questions, ids)):
        question["name"] = f"q{index}"
        names[question["name"]] = qid
    return questions, names


def build_output(data: Dict[str, Any], input_data: AppInput, names: Dict[str, str]) -> AppOutput:
    """The API's response as an AppOutput, with answers under the caller's ids."""
    levels = {q.id: q.levels for q in input_data.scores}
    choices: Dict[str, ChoiceAnswer] = {}
    scores: Dict[str, ScoreAnswer] = {}
    nouls: Dict[str, NoulAnswer] = {}
    for answer in data.get("answers") or []:
        qid = names.get(answer.get("name"))
        if qid is None:
            raise RuntimeError(f"OpenAI returned an answer for an unknown question: {answer.get('name')!r}")
        kind = answer.get("type")
        if kind == "choice":
            choices[qid] = ChoiceAnswer(
                choice=answer["choice"],
                confidence=answer["confidence"],
                probabilities={p["value"]: p["probability"] for p in answer["probabilities"]},
            )
        elif kind == "score":
            scores[qid] = ScoreAnswer(
                score=answer["score"],
                normalized=answer["score"] / (len(levels[qid]) - 1),
                confidence=answer["confidence"],
                probabilities={str(p["value"]): p["probability"] for p in answer["probabilities"]},
                legend={str(i): level for i, level in enumerate(levels[qid])},
            )
        elif kind == "predicate":
            nouls[qid] = NoulAnswer(noul=answer["probability"])
        else:
            raise RuntimeError(f"OpenAI returned an answer of unknown type {kind!r} for question '{qid}'")
    missing = sorted(set(names.values()) - set(choices) - set(scores) - set(nouls))
    if missing:
        raise RuntimeError(f"OpenAI returned no answer for: {missing}")

    input_tokens = int((data.get("usage") or {}).get("input_tokens") or 0)
    return AppOutput(
        choices=choices,
        scores=scores,
        nouls=nouls,
        model=data.get("model") or input_data.model,
        input_tokens=input_tokens,
        output_meta=OutputMeta(inputs=[TextMeta(tokens=input_tokens)], outputs=[TextMeta(tokens=0)]),
    )


# ── App ──────────────────────────────────────────────────────────────────────

def get_api_key() -> str:
    key = os.environ.get("OPENAI_KEY")
    if not key:
        raise RuntimeError("OPENAI_KEY is not set")
    return key.strip()


class App(BaseApp):
    async def setup(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        self.client = httpx.AsyncClient(timeout=120)

    async def _decide(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """One Decisions API call. 429 and 5xx are retried with backoff, honoring retry-after."""
        headers = {"Authorization": f"Bearer {get_api_key()}"}
        for attempt in range(1, MAX_ATTEMPTS + 1):
            resp = await self.client.post(API_URL, headers=headers, json=body)
            if resp.status_code in RETRY_STATUSES and attempt < MAX_ATTEMPTS:
                try:
                    delay = float(resp.headers.get("retry-after", ""))
                except ValueError:
                    delay = min(2 ** (attempt - 1), 30)
                self.logger.info(f"OpenAI returned {resp.status_code}; retry {attempt}/{MAX_ATTEMPTS - 1} in {delay:.0f}s")
                await asyncio.sleep(delay)
                continue
            if resp.status_code == 401:
                raise RuntimeError("OpenAI rejected the API key (401). Check the OPENAI_KEY secret.")
            if resp.status_code >= 400:
                raise RuntimeError(f"OpenAI API error {resp.status_code}: {resp.text[:1000]}")
            return resp.json()
        raise RuntimeError("unreachable")

    async def run(self, input_data: AppInput) -> AppOutput:
        """Evaluate the state and images against every question in one request."""
        questions, names = build_questions(input_data)
        self.logger.info(
            f"model={input_data.model} choices={len(input_data.choices)} scores={len(input_data.scores)} "
            f"nouls={len(input_data.nouls)} images={len(input_data.images)}"
        )
        data = await self._decide({"model": input_data.model, "input": build_input(input_data), "questions": questions})
        output = build_output(data, input_data, names)
        self.logger.info(f"answered by {output.model}: {output.input_tokens} input tokens")
        return output

    async def unload(self):
        await self.client.aclose()

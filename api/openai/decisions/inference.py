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
from typing import Any, Dict, List, Tuple, Union

import httpx
from inferencesh import BaseApp, File
from inferencesh.models.decision import DecisionOutput, DecisionVisionInput, NoulQuestion, Structured
from pydantic import Field

API_URL = "https://api.openai.com/v1/decisions"
MAX_IMAGES = 8
MAX_ATTEMPTS = 4
RETRY_STATUSES = {429, 500, 502, 503, 504}


# ── Input and output ─────────────────────────────────────────────────────────

class AppInput(DecisionVisionInput):
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
    model: str = Field(default="gpt-6-luna", description="The OpenAI model that evaluates the questions. The Decisions API supports `gpt-6-luna` only for now.")


class AppOutput(DecisionOutput):
    input_tokens: int = Field(default=0, description="Input tokens (billed). Image tokens are included.")


# ── Request and response mapping ─────────────────────────────────────────────

def image_data_url(file: File) -> str:
    """An input image as the inline base64 data URL the API requires."""
    if not file.exists():
        raise RuntimeError(f"image does not exist at path: {file.path}")
    mime = file.content_type or mimetypes.guess_type(file.path)[0] or "image/png"
    with open(file.path, "rb") as handle:
        return f"data:{mime};base64,{base64.b64encode(handle.read()).decode()}"


def as_text(value: Structured) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def build_input(input_data: AppInput) -> Union[str, List[Dict[str, Any]]]:
    """The API's `input`: the state as a string, or one user message with the images then the state."""
    text = as_text(input_data.state)
    if not input_data.images:
        return text
    content: List[Dict[str, Any]] = [{"type": "input_image", "image_url": image_data_url(image)} for image in input_data.images]
    if text:
        content.append({"type": "input_text", "text": text})
    return [{"role": "user", "content": content}]


def predicate_instructions(question: NoulQuestion) -> str:
    """A noul's instructions with its criteria appended, since a predicate has no criteria field."""
    criteria = question.criteria.model_dump(exclude_none=True) if question.criteria is not None else {}
    instructions = as_text(question.instructions)
    if not criteria:
        return instructions
    return f"{instructions} ({'; '.join(f'{side}: {as_text(text)}' for side, text in criteria.items())})"


def build_questions(input_data: AppInput) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """The API's `questions`, named q0, q1, ..., and the map from those names back to the caller's ids."""
    questions: List[Dict[str, Any]] = []
    for q in input_data.choices:
        options = [{"value": o.name, **({"description": as_text(o.description)} if o.description else {})} for o in q.options]
        questions.append({"type": "choice", "instructions": as_text(q.instructions), "choices": options})
    for q in input_data.scores:
        questions.append({"type": "score", "instructions": as_text(q.instructions), "levels": [{"label": as_text(level)} for level in q.levels]})
    for q in input_data.nouls:
        questions.append({"type": "predicate", "instructions": predicate_instructions(q)})
    ids = [q.id for q in (*input_data.choices, *input_data.scores, *input_data.nouls)]
    names: Dict[str, str] = {}
    for index, (question, qid) in enumerate(zip(questions, ids)):
        question["name"] = f"q{index}"
        names[question["name"]] = qid
    return questions, names


def build_answers(data: Dict[str, Any], input_data: AppInput, names: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """The API's answers in the decision contract's form, under the caller's ids."""
    levels = {q.id: q.levels for q in input_data.scores}
    answers: Dict[str, Dict[str, Any]] = {}
    for answer in data.get("answers") or []:
        qid = names.get(answer.get("name"))
        if qid is None:
            raise RuntimeError(f"OpenAI returned an answer for an unknown question: {answer.get('name')!r}")
        kind = answer.get("type")
        if kind == "choice":
            answers[qid] = {
                "choice": answer["choice"],
                "confidence": answer["confidence"],
                "probabilities": {p["value"]: p["probability"] for p in answer["probabilities"]},
            }
        elif kind == "score":
            answers[qid] = {
                "score": answer["score"],
                "confidence": answer["confidence"],
                "probabilities": {str(p["value"]): p["probability"] for p in answer["probabilities"]},
                "legend": {str(i): level for i, level in enumerate(levels[qid])},
            }
        elif kind == "predicate":
            answers[qid] = {"noul": answer["probability"]}
        else:
            raise RuntimeError(f"OpenAI returned an answer of unknown type {kind!r} for question '{qid}'")
    return answers


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

    async def _decide(self, body: bytes) -> Dict[str, Any]:
        """One Decisions API call. 429 and 5xx are retried with backoff, honoring retry-after."""
        headers = {"Authorization": f"Bearer {get_api_key()}", "Content-Type": "application/json"}
        for attempt in range(1, MAX_ATTEMPTS + 1):
            resp = await self.client.post(API_URL, headers=headers, content=body)
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
        # Reading and encoding images blocks, and the body with them is megabytes: build it once, off the loop.
        body = await asyncio.to_thread(
            lambda: json.dumps({"model": input_data.model, "input": build_input(input_data), "questions": questions}).encode()
        )
        data = await self._decide(body)
        input_tokens = int((data.get("usage") or {}).get("input_tokens") or 0)
        model = data.get("model") or input_data.model
        self.logger.info(f"answered by {model}: {input_tokens} input tokens")
        return AppOutput.from_answers(build_answers(data, input_data, names), input_data, model=model, input_tokens=input_tokens)

    async def unload(self):
        await self.client.aclose()

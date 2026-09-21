"""
Jev — TypeSafe's System One model.

One request evaluates one `state` against any number of typed questions and returns a
typed, calibrated answer for each. Three question types (the primitives):

  choice  which of these options?   -> choice, probabilities, confidence
  score   which level on a scale?   -> score, probabilities, confidence, legend
  noul    is this true?             -> noul (probability of yes, 0 to 1)

The TypeSafe API takes one `questions` map keyed by id, discriminated by `type`. Here each
primitive has its own typed list (`choices`, `scores`, `nouls`) and every question carries
its `id`; answers come back keyed by the same ids. Ids share one namespace.

`state`, `instructions` and every option / level / criteria description accept a string,
an object or an array: Jev is trained to read JSON structure.

API: POST https://api.typesafe.ai/v1/systemone — https://docs.typesafe.ai/api
"""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, Union

import httpx
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, OutputMeta, TextMeta
from pydantic import BaseModel, Field, model_validator

API_BASE = "https://api.typesafe.ai/v1"
RETRY_STATUSES = (429, 529)
MAX_ATTEMPTS = 6

# TypeSafe's EntryType: plain text, or JSON structure the model reads by key.
Structured = Union[str, Dict[str, Any], List[Any]]


# ── Questions ────────────────────────────────────────────────────────────────

class ChoiceOption(BaseModel):
    name: str = Field(description="Option name. Returned as `choice` and used as the key in `probabilities`. Sent to the model.")
    description: Optional[Structured] = Field(
        default=None,
        description="What this option covers. Omit when the name is clear on its own. An object can carry a rubric, e.g. {what, not_for, examples}, or a taxonomy subtree.",
    )


class ChoiceQuestion(BaseModel):
    """Which of these options? For a fixed set of unordered options."""
    id: str = Field(description="Your key for this question; the answer comes back under it. Not sent to the model.")
    instructions: Structured = Field(
        description="What the model should decide, written as a complete question. Reference parts of a structured state by path in backticks, e.g. `ticket.messages[0].text`. An object can hold the question in one field and data it refers to in others.",
    )
    options: List[ChoiceOption] = Field(
        min_length=2,
        max_length=255,
        description="The answer options (2 to 255). Give the full list, and add an `other` option when the list might not cover every input.",
    )


class ScoreQuestion(BaseModel):
    """Which level? For a position on a spectrum you can describe."""
    id: str = Field(description="Your key for this question; the answer comes back under it. Not sent to the model.")
    instructions: Structured = Field(
        description="What the model should rate, written as a complete question. Reference parts of a structured state by path in backticks.",
    )
    levels: List[Structured] = Field(
        min_length=2,
        max_length=10,
        description="Ordered level descriptions, low end to high end (2 to 10). A level's number is its index, starting at 0. Describe situations, not degrees: each level is judged on its own and the model never sees its number or its neighbours.",
    )


class NoulCriteria(BaseModel):
    true: Optional[Structured] = Field(default=None, description="What a yes (value near 1) means.")
    false: Optional[Structured] = Field(default=None, description="What a no (value near 0) means.")


class NoulQuestion(BaseModel):
    """Is this true? For a clean yes/no where the probability itself is the signal."""
    id: str = Field(description="Your key for this question; the answer comes back under it. Not sent to the model.")
    instructions: Structured = Field(
        description="The yes/no question, or a statement to judge. Make the boundary between yes and no unambiguous. Reference parts of a structured state by path in backticks.",
    )
    criteria: Optional[NoulCriteria] = Field(
        default=None,
        description="Optional. Pins down a subtle yes/no boundary.",
    )


class AppInput(BaseAppInput):
    state: Structured = Field(
        description="The content to evaluate: a string, or a JSON object / array of related context (messages, records, a policy). Text only. Every question sees the same state. Send only what the questions need; unrelated detail costs accuracy.",
        examples=["Our API started returning 500 errors 20 minutes ago and we can't process any customer orders."],
    )
    choices: List[ChoiceQuestion] = Field(default_factory=list, description="Choice questions: pick one option from a set.")
    scores: List[ScoreQuestion] = Field(default_factory=list, description="Score questions: place the state on ordered levels.")
    nouls: List[NoulQuestion] = Field(default_factory=list, description="Noul questions: probability that the answer is yes.")
    model: str = Field(
        default="jev-latest",
        description="Model name or alias: `jev-latest`, `jev-preview`, or a pinned version such as `jev-1.13.0`. Pin a version if you tuned thresholds against it.",
    )

    @model_validator(mode="after")
    def _check_questions(self):
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
    model: str = Field(description="The versioned model that answered, e.g. `jev-1.13.0`.")
    input_tokens: int = Field(default=0, description="Input tokens (billed).")
    output_tokens: int = Field(default=0, description="Output tokens (free upstream).")


# ── list_models ──────────────────────────────────────────────────────────────

class ListModelsInput(BaseAppInput):
    pass


class ModelCard(BaseModel):
    name: str = Field(description="Model id or alias, as accepted by `model`.")
    description: str = Field(default="", description="What the model is for.")
    release_date: str = Field(default="", description="When the model or alias was released.")


class ListModelsOutput(BaseAppOutput):
    models: List[ModelCard] = Field(description="Names this account can send in `model`. Pinned versions are accepted even when not listed.")


# ── App ──────────────────────────────────────────────────────────────────────

def get_api_key() -> str:
    key = os.environ.get("TYPESAFE_KEY")
    if not key:
        raise RuntimeError(
            "TYPESAFE_KEY is not set. A secret whose record exists but holds an empty "
            "value is not injected at all — check that `belt secrets get TYPESAFE_KEY "
            "--json` reports a non-empty masked_value, and re-set it if it does not."
        )
    return key.strip()


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


class App(BaseApp):
    async def setup(self, metadata):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        self.client = httpx.AsyncClient(base_url=API_BASE, timeout=120)
        self._cancel = False

    async def on_cancel(self):
        self._cancel = True
        return True

    def _cancelled(self) -> bool:
        ctx = getattr(self, "context", None)
        return self._cancel or (ctx is not None and ctx.cancel_requested)

    async def _request(self, method: str, path: str, **kwargs) -> Dict[str, Any]:
        """One API call. 429 / 529 are retried with backoff, honoring retry-after."""
        headers = {"Authorization": f"Bearer {get_api_key()}"}
        for attempt in range(1, MAX_ATTEMPTS + 1):
            resp = await self.client.request(method, path, headers=headers, **kwargs)
            if resp.status_code in RETRY_STATUSES and attempt < MAX_ATTEMPTS and not self._cancelled():
                try:
                    delay = float(resp.headers.get("retry-after", ""))
                except ValueError:
                    delay = min(2 ** (attempt - 1), 30)
                self.logger.info(f"TypeSafe returned {resp.status_code}; retry {attempt}/{MAX_ATTEMPTS - 1} in {delay:.0f}s")
                await asyncio.sleep(delay)
                continue
            if resp.status_code == 401:
                raise RuntimeError("TypeSafe rejected the API key (401). Check the TYPESAFE_KEY secret.")
            if resp.status_code >= 400:
                raise RuntimeError(f"TypeSafe API error {resp.status_code}: {resp.text[:1000]}")
            return resp.json()
        raise RuntimeError("unreachable")

    async def run(self, input_data: AppInput) -> AppOutput:
        """Evaluate the state against every question in one request."""
        self._cancel = False
        questions = build_questions(input_data)
        self.logger.info(
            f"model={input_data.model} choices={len(input_data.choices)} "
            f"scores={len(input_data.scores)} nouls={len(input_data.nouls)}"
        )

        data = await self._request(
            "POST", "/systemone",
            json={"state": input_data.state, "model": input_data.model, "questions": questions},
        )

        answers = data.get("answers") or {}
        top_level = {q.id: len(q.levels) - 1 for q in input_data.scores}
        choices: Dict[str, ChoiceAnswer] = {}
        scores: Dict[str, ScoreAnswer] = {}
        nouls: Dict[str, NoulAnswer] = {}
        for qid, answer in answers.items():
            kind = answer.get("type")
            if kind == "choice":
                choices[qid] = ChoiceAnswer(
                    choice=answer.get("choice", ""),
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
                self.logger.warning(f"answer '{qid}' has unknown type {kind!r}; keys: {list(answer.keys())}")

        missing = sorted(set(questions) - set(answers))
        if missing:
            raise RuntimeError(f"TypeSafe returned no answer for: {missing}")

        usage = data.get("usage") or {}
        input_tokens = int(usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
        self.logger.info(f"answered by {data.get('model')}: {input_tokens} in / {output_tokens} out tokens")

        return AppOutput(
            choices=choices,
            scores=scores,
            nouls=nouls,
            model=data.get("model") or input_data.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            output_meta=OutputMeta(
                inputs=[TextMeta(tokens=input_tokens)],
                outputs=[TextMeta(tokens=output_tokens)],
            ),
        )

    async def list_models(self, input_data: ListModelsInput) -> ListModelsOutput:
        """List the model names and aliases this account can use. Free."""
        data = await self._request("GET", "/models")
        return ListModelsOutput(
            models=[
                ModelCard(
                    name=m.get("name", ""),
                    description=m.get("description") or "",
                    release_date=str(m.get("release_date") or ""),
                )
                for m in data.get("models") or []
            ],
            output_meta=OutputMeta(inputs=[], outputs=[]),
        )

    async def unload(self):
        await self.client.aclose()

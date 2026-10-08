"""
JEV-27B-VL — AutoTrust's open decision model with vision, run on our own GPUs.

One request evaluates one `state` (text, plus up to 8 images) against any number of typed
questions and returns a probability for every option. No text is generated. Three
question types:

  choice  which of these options?   -> choice, probabilities, confidence
  score   which level on a scale?   -> score, probabilities, confidence, legend
  noul    is this true?             -> noul (probability of yes, 0 to 1)

The input and output match `typesafe/jev` and the `decision-2-0-*` apps, with `images`
added, so a caller can move between them.

Serving follows the model card: the repository's `serve_decide.py` is the vLLM OpenAI
server plus `POST /v1/decide`, which builds the decision prompt, reads the option tokens
and applies the decision head's bias and temperature. setup() starts it as a child
process on localhost; run() sends one /v1/decide call per question, concurrently. The
server answers one question per call; the state is shared through vLLM's prefix cache.

How the typed questions map to /v1/decide:

  choice  kind=choice; each option is one line, `name` or `name: description`
  noul    kind=noul; `criteria` is appended to the question
  score   six levels use the model's native 0-5 score head, with the levels listed in the
          question; any other number of levels is read as a choice over the levels

Metering: inputs=[TextMeta(tokens=<prompt tokens, summed over questions>)], outputs=[TextMeta(tokens=0)].

Model: https://huggingface.co/autotrust/JEV-27B-VL
"""

import asyncio
import base64
import json
import logging
import math
import mimetypes
import socket
import sys
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, TextMeta
from pydantic import BaseModel, Field, model_validator

MODEL_ID = "autotrust/JEV-27B-VL"
# setup() executes the repository's serve_decide.py. Review the diff before moving this.
REVISION = "f34b598d4ef4bcefd337bee8d8e7ddd3b7733ccc"
# Prompt tokens per question: state, images, question and options. The backbone takes up
# to 262,144, at about 65 KB of KV cache per token on top of 52 GB of weights.
MAX_MODEL_LEN = 32768
MAX_IMAGES = 8
# Required by the model card: above 8 sequences in a batch, vLLM's LoRA path for this
# multimodal model class returns wrong probabilities. Further requests queue.
MAX_NUM_SEQS = 8
NATIVE_SCORE_LEVELS = 6
SERVER_LOG_TAIL = 40

# Plain text, or JSON structure the model reads by key.
Structured = Union[str, Dict[str, Any], List[Any]]


# ── Questions ────────────────────────────────────────────────────────────────

class ChoiceOption(BaseModel):
    name: str = Field(min_length=1, description="Option name. Returned as `choice` and used as the key in `probabilities`. Sent to the model.")
    description: Optional[Structured] = Field(
        default=None,
        description="What this option covers, sent on the option's line after the name. One line saying what separates it from similar options works best. Omit when the name is clear on its own.",
    )


class ChoiceQuestion(BaseModel):
    """Which of these options? For a fixed set of unordered options."""
    id: str = Field(min_length=1, description="Your key for this question; the answer comes back under it. Not sent to the model.")
    instructions: Structured = Field(description="What the model should decide, written as a complete question.")
    options: List[ChoiceOption] = Field(
        min_length=2,
        max_length=255,
        description="The answer options (2 to 255). The model was trained on up to 16; more are read with untrained labels. Add an `other` option when the list might not cover every input.",
    )


class ScoreQuestion(BaseModel):
    """Which level? For a position on a spectrum you can describe."""
    id: str = Field(min_length=1, description="Your key for this question; the answer comes back under it. Not sent to the model.")
    instructions: Structured = Field(description="What the model should rate, written as a complete question.")
    levels: List[Structured] = Field(
        min_length=2,
        max_length=10,
        description="Ordered level descriptions, low end to high end (2 to 10). A level's number is its index, starting at 0. Six levels use the model's native 0-5 score; any other number is read as a choice over the levels.",
    )


class NoulCriteria(BaseModel):
    true: Optional[Structured] = Field(default=None, description="What a yes (value near 1) means.")
    false: Optional[Structured] = Field(default=None, description="What a no (value near 0) means.")


class NoulQuestion(BaseModel):
    """Is this true? For a clean yes/no where the probability itself is the signal."""
    id: str = Field(min_length=1, description="Your key for this question; the answer comes back under it. Not sent to the model.")
    instructions: Structured = Field(
        description="The yes/no question, phrased so that yes is the outcome you want the probability of, e.g. `Does the image show food or cooking?`",
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
        description="Up to 8 images the questions are about, placed before the state. Large images cost more tokens and time; about 448 px on the long side is enough for most decisions.",
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
    confidence: float = Field(description="0 to 1, from how peaked `probabilities` is: 1 minus its entropy over the maximum entropy. Gate actions on it; thresholds scale with risk.")
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
    model: str = Field(description="The model that answered: `autotrust/JEV-27B-VL`.")
    input_tokens: int = Field(default=0, description="Prompt tokens, summed over the questions. Image tokens are included.")


# ── Request building ─────────────────────────────────────────────────────────

def as_text(value: Structured) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def as_line(value: Structured) -> str:
    """The decision template lists one option per line, so option text carries no newlines."""
    return " ".join(as_text(value).split())


def image_part(image: File) -> Dict[str, str]:
    content_type = image.content_type or mimetypes.guess_type(image.path)[0] or "image/jpeg"
    with open(image.path, "rb") as f:
        return {"image": f"data:{content_type};base64,{base64.b64encode(f.read()).decode()}"}


def build_state(input_data: AppInput) -> Union[str, List[Any]]:
    text = as_text(input_data.state)
    if not input_data.images:
        return text
    parts: List[Any] = [image_part(image) for image in input_data.images]
    if text:
        parts.append("\n" + text)
    return parts


def build_requests(input_data: AppInput) -> List[Tuple[str, str, Dict[str, Any]]]:
    """One (question id, question type, /v1/decide body without state) per question."""
    requests: List[Tuple[str, str, Dict[str, Any]]] = []
    for q in input_data.choices:
        options = [as_line(o.name) if o.description is None else f"{as_line(o.name)}: {as_line(o.description)}" for o in q.options]
        requests.append((q.id, "choice", {"kind": "choice", "question": as_text(q.instructions), "options": options}))
    for q in input_data.scores:
        levels = [as_line(level) for level in q.levels]
        if len(levels) == NATIVE_SCORE_LEVELS:
            legend = "; ".join(f"{i} = {level}" for i, level in enumerate(levels))
            body = {"kind": "score", "question": f"{as_text(q.instructions)} Levels: {legend}"}
        else:
            body = {"kind": "choice", "question": as_text(q.instructions), "options": [f"{i}: {level}" for i, level in enumerate(levels)]}
        requests.append((q.id, "score", body))
    for q in input_data.nouls:
        question = as_text(q.instructions)
        criteria = q.criteria.model_dump(exclude_none=True) if q.criteria else {}
        if criteria:
            question += " (" + "; ".join(f"{key}: {as_line(criteria[key])}" for key in ("true", "false") if key in criteria) + ")"
        requests.append((q.id, "noul", {"kind": "noul", "question": question}))
    return requests


def confidence(probabilities: List[float]) -> float:
    entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
    return max(0.0, 1.0 - entropy / math.log(len(probabilities)))


def build_output(input_data: AppInput, results: Dict[str, Dict[str, Any]]) -> AppOutput:
    choices: Dict[str, ChoiceAnswer] = {}
    scores: Dict[str, ScoreAnswer] = {}
    nouls: Dict[str, NoulAnswer] = {}
    for q in input_data.choices:
        probabilities = results[q.id]["probabilities"]
        names = [o.name for o in q.options]
        choices[q.id] = ChoiceAnswer(
            choice=names[max(range(len(names)), key=probabilities.__getitem__)],
            confidence=confidence(probabilities),
            probabilities=dict(zip(names, probabilities)),
        )
    for q in input_data.scores:
        probabilities = results[q.id]["probabilities"]
        score = sum(i * p for i, p in enumerate(probabilities))
        scores[q.id] = ScoreAnswer(
            score=score,
            normalized=score / (len(q.levels) - 1),
            confidence=confidence(probabilities),
            probabilities={str(i): p for i, p in enumerate(probabilities)},
            legend={str(i): level for i, level in enumerate(q.levels)},
        )
    for q in input_data.nouls:
        # /v1/decide returns noul probabilities for ["false", "true"].
        nouls[q.id] = NoulAnswer(noul=results[q.id]["probabilities"][1])

    input_tokens = sum(int((r.get("usage") or {}).get("prompt_tokens") or 0) for r in results.values())
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


def expected_options(input_data: AppInput) -> Dict[str, int]:
    counts = {q.id: len(q.options) for q in input_data.choices}
    counts.update({q.id: len(q.levels) for q in input_data.scores})
    counts.update({q.id: 2 for q in input_data.nouls})
    return counts


# ── App ──────────────────────────────────────────────────────────────────────

class App(BaseApp):
    async def setup(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        logging.getLogger("httpx").setLevel(logging.WARNING)

        from huggingface_hub import snapshot_download

        started = time.monotonic()
        self.logger.info(f"downloading {MODEL_ID}@{REVISION[:8]}")
        model_dir = await asyncio.to_thread(snapshot_download, MODEL_ID, revision=REVISION)
        self.logger.info(f"downloaded in {time.monotonic() - started:.1f}s")

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        self.base_url = f"http://127.0.0.1:{port}"
        self.server_log: deque = deque(maxlen=SERVER_LOG_TAIL)
        self.server = await asyncio.create_subprocess_exec(
            sys.executable, f"{model_dir}/serve_decide.py",
            "--model", model_dir,
            "--served-model-name", MODEL_ID,
            "--enable-lora", "--max-lora-rank", "32",
            "--lora-modules", f"jev-decision={model_dir}/adapter_vllm",
            "--logprobs-mode", "processed_logprobs",
            "--max-model-len", str(MAX_MODEL_LEN),
            "--enable-prefix-caching", "--mamba-cache-mode", "align",
            "--limit-mm-per-prompt", json.dumps({"image": MAX_IMAGES}),
            "--max-num-seqs", str(MAX_NUM_SEQS),
            "--trust-request-chat-template",
            "--host", "127.0.0.1", "--port", str(port),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        self.server_log_task = asyncio.create_task(self._forward_server_log())
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=None)

        # No deadline here: the platform owns the setup timeout. Stop only if the server exits.
        while True:
            if self.server.returncode is not None:
                raise RuntimeError(f"the vLLM server exited with code {self.server.returncode} during startup:\n" + "\n".join(self.server_log))
            try:
                if (await self.client.get("/health")).status_code == 200:
                    break
            except httpx.TransportError:
                pass
            await asyncio.sleep(2)
        info = (await self.client.get("/v1/decide/info")).json()
        self.logger.info(
            f"vLLM server ready in {time.monotonic() - started:.1f}s; up to {info.get('max_options')} options, "
            f"{MAX_MODEL_LEN} prompt tokens, temperatures {info.get('temperatures')}"
        )

    async def _forward_server_log(self):
        async for raw in self.server.stdout:
            line = raw.decode(errors="replace").rstrip()
            if line:
                self.server_log.append(line)
                self.logger.info(f"[vllm] {line}")

    async def _decide(self, qid: str, state: Union[str, List[Any]], body: Dict[str, Any]) -> Dict[str, Any]:
        response = await self.client.post("/v1/decide", json={"state": state, **body})
        if response.status_code != 200:
            try:
                error = response.json().get("error") or {}
                message = error.get("message") or response.text
            except ValueError:
                message = response.text
            raise RuntimeError(f"question '{qid}' could not be answered ({response.status_code}): {message[:600]}")
        return response.json()

    async def run(self, input_data: AppInput) -> AppOutput:
        """Evaluate the state and images against every question."""
        if self.server.returncode is not None:
            raise RuntimeError(f"the vLLM server exited with code {self.server.returncode}:\n" + "\n".join(self.server_log))
        for image in input_data.images:
            if not image.exists():
                raise RuntimeError(f"image does not exist at path: {image.path}")

        state = await asyncio.to_thread(build_state, input_data)
        requests = build_requests(input_data)
        started = time.monotonic()
        answers = await asyncio.gather(*(self._decide(qid, state, body) for qid, _, body in requests))
        elapsed_ms = (time.monotonic() - started) * 1000

        results = {qid: answer for (qid, _, _), answer in zip(requests, answers)}
        for qid, count in expected_options(input_data).items():
            got = results[qid].get("probabilities") or []
            if len(got) != count:
                raise RuntimeError(f"question '{qid}' came back with {len(got)} probabilities for {count} options; keys: {list(results[qid].keys())}")

        output = build_output(input_data, results)
        self.logger.info(
            f"answered by {MODEL_ID}: {output.input_tokens} prompt tokens, {len(requests)} questions, "
            f"{len(input_data.images)} images, {elapsed_ms:.1f} ms"
        )
        return output

    async def unload(self):
        if hasattr(self, "client"):
            await self.client.aclose()
        if hasattr(self, "server") and self.server.returncode is None:
            self.server.terminate()
            await self.server.wait()
        if hasattr(self, "server_log_task"):
            self.server_log_task.cancel()

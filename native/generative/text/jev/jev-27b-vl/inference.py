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
from typing import Any, Dict, List, Tuple

import httpx
from inferencesh import BaseApp, File
from inferencesh.models.decision import DecisionOutput, DecisionVisionInput, Structured
from pydantic import Field

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
        description="Up to 8 images the questions are about, placed before the state. Large images cost more tokens and time; about 448 px on the long side is enough for most decisions.",
    )


class AppOutput(DecisionOutput):
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


def encode_state(input_data: AppInput) -> bytes:
    """The /v1/decide `state` as JSON: the text, or the images then the text. Every question's
    request carries it, so it is encoded once here; with images it is megabytes of base64."""
    text = as_text(input_data.state)
    if not input_data.images:
        return json.dumps(text).encode()
    parts: List[Any] = [image_part(image) for image in input_data.images]
    if text:
        parts.append("\n" + text)
    return json.dumps(parts).encode()


def build_requests(input_data: AppInput) -> List[Tuple[str, Dict[str, Any], int]]:
    """One (question id, /v1/decide body without state, probabilities expected back) per question."""
    requests: List[Tuple[str, Dict[str, Any], int]] = []
    for q in input_data.choices:
        options = [as_line(o.name) if o.description is None else f"{as_line(o.name)}: {as_line(o.description)}" for o in q.options]
        requests.append((q.id, {"kind": "choice", "question": as_text(q.instructions), "options": options}, len(options)))
    for q in input_data.scores:
        levels = [as_line(level) for level in q.levels]
        if len(levels) == NATIVE_SCORE_LEVELS:
            legend = "; ".join(f"{i} = {level}" for i, level in enumerate(levels))
            body = {"kind": "score", "question": f"{as_text(q.instructions)} Levels: {legend}"}
        else:
            body = {"kind": "choice", "question": as_text(q.instructions), "options": [f"{i}: {level}" for i, level in enumerate(levels)]}
        requests.append((q.id, body, len(levels)))
    for q in input_data.nouls:
        question = as_text(q.instructions)
        criteria = q.criteria.model_dump(exclude_none=True) if q.criteria else {}
        if criteria:
            question += " (" + "; ".join(f"{key}: {as_line(criteria[key])}" for key in ("true", "false") if key in criteria) + ")"
        requests.append((q.id, {"kind": "noul", "question": question}, 2))
    return requests


def confidence(probabilities: List[float]) -> float:
    entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
    return max(0.0, 1.0 - entropy / math.log(len(probabilities)))


def build_answers(input_data: AppInput, probabilities: Dict[str, List[float]]) -> Dict[str, Dict[str, Any]]:
    """Each question's probabilities, in option order, as an answer in the decision contract's form."""
    answers: Dict[str, Dict[str, Any]] = {}
    for q in input_data.choices:
        values = probabilities[q.id]
        names = [o.name for o in q.options]
        answers[q.id] = {
            "choice": names[max(range(len(names)), key=values.__getitem__)],
            "confidence": confidence(values),
            "probabilities": dict(zip(names, values)),
        }
    for q in input_data.scores:
        values = probabilities[q.id]
        answers[q.id] = {
            "score": sum(i * p for i, p in enumerate(values)),
            "confidence": confidence(values),
            "probabilities": {str(i): p for i, p in enumerate(values)},
            "legend": {str(i): level for i, level in enumerate(q.levels)},
        }
    for q in input_data.nouls:
        # /v1/decide returns noul probabilities for ["false", "true"].
        answers[q.id] = {"noul": probabilities[q.id][1]}
    return answers


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

    async def _decide(self, qid: str, state: bytes, body: Dict[str, Any]) -> Dict[str, Any]:
        # The body is the question's fields with the already encoded state spliced in first.
        content = b'{"state":' + state + b"," + json.dumps(body).encode()[1:]
        response = await self.client.post("/v1/decide", content=content, headers={"Content-Type": "application/json"})
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

        state = await asyncio.to_thread(encode_state, input_data)
        requests = build_requests(input_data)
        started = time.monotonic()
        results = await asyncio.gather(*(self._decide(qid, state, body) for qid, body, _ in requests))
        elapsed_ms = (time.monotonic() - started) * 1000

        probabilities: Dict[str, List[float]] = {}
        for (qid, _, expected), result in zip(requests, results):
            probabilities[qid] = result.get("probabilities") or []
            if len(probabilities[qid]) != expected:
                raise RuntimeError(f"question '{qid}' came back with {len(probabilities[qid])} probabilities for {expected} options; keys: {list(result.keys())}")
        input_tokens = sum(int((result.get("usage") or {}).get("prompt_tokens") or 0) for result in results)

        self.logger.info(
            f"answered by {MODEL_ID}: {input_tokens} prompt tokens, {len(requests)} questions, "
            f"{len(input_data.images)} images, {elapsed_ms:.1f} ms"
        )
        return AppOutput.from_answers(build_answers(input_data, probabilities), input_data, model=MODEL_ID, input_tokens=input_tokens)

    async def unload(self):
        if hasattr(self, "client"):
            await self.client.aclose()
        if hasattr(self, "server") and self.server.returncode is None:
            self.server.terminate()
            await self.server.wait()
        if hasattr(self, "server_log_task"):
            self.server_log_task.cancel()

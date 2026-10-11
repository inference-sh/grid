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
server answers one question per call.

Sharing the state between questions: the model is a hybrid (attention + mamba layers), so vLLM
raises its cache block to 784 tokens and in `align` mode only resumes a cached prompt where a mamba
state was saved. A 448 px image and a question are ~165 tokens, so questions never shared
anything. With --prefix-match-unit 16, vLLM also saves the state at each prompt's tail (rounded
down to 16 tokens). When a request's questions share enough prefix (decide_warmups), run() first
sends a warm-up: serve_decide's prompt up to where the question starts. Every question then
resumes from that state. The questions' inputs are unchanged; answers move within ~0.01 of an
uncached run, because the cached and recomputed mamba states round differently in bf16.

Weights come from inference-sh/JEV-27B-VL-merged: the model with its decision LoRA
(`adapter_vllm`, rank 32 on every language-model linear and lm_head) folded into the base
weights. Served as a LoRA, vLLM's LoRA kernels added about 40% to a short decision. That repo
also carries the decision head, the calibration and serve_decide.py, patched to take the head
from JEV_DECIDE_ADAPTER_DIR and to send System 1 to the served model (JEV_DECIDE_LORA). Only
System 1 is used (thinking is off), so the unmerged base model is never needed.

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
import os
import socket
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx
from inferencesh import BaseApp, File
from inferencesh.models.decision import DecisionOutput, DecisionVisionInput, Structured
from pydantic import Field

MODEL_ID = "autotrust/JEV-27B-VL"
# autotrust/JEV-27B-VL f34b598d with the decision LoRA merged. setup() executes the repository's
# serve_decide.py. Review the diff before moving this.
WEIGHTS_ID = "inference-sh/JEV-27B-VL-merged"
WEIGHTS_REVISION = "0b8292a7eefb323b13e32a6fdecd70d9f8b39884"
# Prompt tokens per question: state, images, question and options. The backbone takes up
# to 262,144, at about 65 KB of KV cache per token on top of 52 GB of weights.
MAX_MODEL_LEN = 32768
MAX_IMAGES = 8
# The model card requires 8: above 8 sequences in a batch, vLLM's LoRA path for this
# multimodal model class returns wrong probabilities. The merged weights skip that path. Checked
# on 13 request shapes (bench app, cache reset per call): batch 16 answers match batch 8 within
# 0.0021, inside the 0.0033 two identical runs already differ by, and 16 questions on one image
# run in one batch instead of two (~20% faster). Further requests queue.
MAX_NUM_SEQS = 16
NATIVE_SCORE_LEVELS = 6
# Prefix-cache hits can land every 16 tokens inside a 784-token hybrid block, and align mode
# checkpoints the mamba state at each prompt's tail (rounded down to 16). A prompt that is only the
# shared image prefix therefore leaves a state every question on that image can resume from.
PREFIX_MATCH_UNIT = 16
WARM_READ = {"max_tokens": 1, "temperature": 0, "add_special_tokens": False}
# serve_decide.py's raw template for the qwen profile: content parts rendered with no chat wrapper.
RAW_TEMPLATE = ("{%- for m in messages -%}{%- if m['content'] is string -%}{{ m['content'] }}{%- else -%}"
                "{%- for c in m['content'] -%}{%- if c['type'] == 'text' -%}{{ c['text'] }}{%- else -%}"
                "<|vision_start|><|image_pad|><|vision_end|>{%- endif -%}{%- endfor -%}{%- endif -%}{%- endfor -%}")
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


def encode_state(image_urls: List[str], text: str) -> bytes:
    """The /v1/decide `state` as JSON: the text, or the images then the text. Every question's
    request carries it, so it is encoded once here; with images it is megabytes of base64."""
    if not image_urls:
        return json.dumps(text).encode()
    parts: List[Any] = [{"image": url} for url in image_urls]
    if text:
        parts.append("\n" + text)
    return json.dumps(parts).encode()


def warm_request(kind: str, image_urls: List[str], text: str) -> Tuple[str, Dict[str, Any]]:
    """The request that prefills only what every `kind` question on this state shares: serve_decide's
    prompt up to where the question starts. Mirrors s1_pass: chat parts with the raw template when
    there are images, a plain completion over the joined text otherwise."""
    head = f"[kind] {kind}\n[state] "
    if not image_urls:
        return "/v1/completions", {"model": MODEL_ID, "prompt": head + text, **WARM_READ}
    content: List[Dict[str, Any]] = [{"type": "text", "text": head}]
    content += [{"type": "image_url", "image_url": {"url": url}} for url in image_urls]
    if text:
        content.append({"type": "text", "text": "\n" + text})
    return "/v1/chat/completions", {"model": MODEL_ID, "messages": [{"role": "user", "content": content}],
                                    "chat_template": RAW_TEMPLATE, "add_generation_prompt": False, **WARM_READ}


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


# ── Warm-up gate ─────────────────────────────────────────────────────────────
#
# A warm-up is one extra request that runs before the questions; it saves every other question of
# that kind the prefill of the shared prefix. Questions resume from the warm-up's state at its last
# PREFIX_MATCH_UNIT boundary, so each saves floor16(prefix) tokens. Measured on A100 (bench app,
# 13 request shapes, cache reset per call): warm-ups lost at <= 384 saved tokens and with 2
# questions, and won from 640 (1.12x, an image with 6 questions) to 3.4x (16 questions on ~1,300
# tokens of text).
WARM_MIN_QUESTIONS = 3
WARM_MIN_SAVED_TOKENS = 600
# English measured at ~4.6 characters per token. Scripts with fewer characters per token are
# underestimated, which only means fewer warm-ups.
CHARS_PER_TOKEN = 4.5
# "[kind] noul\n[state] " plus the vision start/end markers around each image.
TEMPLATE_TOKENS = 8
VISION_MARKER_TOKENS = 2


@dataclass(frozen=True)
class ImageGrid:
    """How the processor turns pixels into tokens: images are resized to a multiple of `factor`
    (patch size x merge size) within [min_pixels, max_pixels], one token per factor x factor."""
    factor: int = 32
    min_pixels: int = 65536
    max_pixels: int = 16777216

    @classmethod
    def from_preprocessor_config(cls, path: str) -> "ImageGrid":
        try:
            with open(path) as f:
                config = json.load(f)
        except (OSError, ValueError):
            return cls()
        size = config.get("size") or {}
        return cls(
            factor=int(config.get("patch_size", 16)) * int(config.get("merge_size", 2)),
            min_pixels=int(size.get("shortest_edge", config.get("min_pixels", cls.min_pixels))),
            max_pixels=int(size.get("longest_edge", config.get("max_pixels", cls.max_pixels))),
        )

    def tokens(self, width: int, height: int) -> int:
        """Qwen's smart_resize, then one token per grid cell."""
        f = self.factor
        h = max(f, round(height / f) * f)
        w = max(f, round(width / f) * f)
        if h * w > self.max_pixels:
            beta = math.sqrt(height * width / self.max_pixels)
            h = max(f, math.floor(height / beta / f) * f)
            w = max(f, math.floor(width / beta / f) * f)
        elif h * w < self.min_pixels:
            beta = math.sqrt(self.min_pixels / (height * width))
            h = math.ceil(height * beta / f) * f
            w = math.ceil(width * beta / f) * f
        return (h // f) * (w // f)


@dataclass(frozen=True)
class WarmupDecision:
    kind: str
    questions: int
    prefix_tokens: int
    saved_tokens: int
    warm: bool


def estimate_prefix_tokens(image_sizes: List[Tuple[int, int]], text: str, grid: ImageGrid) -> int:
    """Tokens every question of one kind shares: the template head, the images and the state text."""
    images = sum(grid.tokens(w, h) + VISION_MARKER_TOKENS for w, h in image_sizes)
    return TEMPLATE_TOKENS + images + math.ceil(len(text) / CHARS_PER_TOKEN)


def saved_tokens(questions: int, prefix_tokens: int) -> int:
    """Prefill a warm-up saves: each question after the first resumes at the last match boundary."""
    return max(0, questions - 1) * (prefix_tokens // PREFIX_MATCH_UNIT * PREFIX_MATCH_UNIT)


def decide_warmups(image_sizes: List[Tuple[int, int]], text: str, kinds: List[str], grid: ImageGrid,
                   min_questions: int = WARM_MIN_QUESTIONS, min_saved_tokens: int = WARM_MIN_SAVED_TOKENS) -> List[WarmupDecision]:
    """Per question kind: warm up when the prefill it saves the other questions outweighs the extra
    request that runs first. Reads only shapes: image sizes, text length and question counts."""
    prefix = estimate_prefix_tokens(image_sizes, text, grid)
    decisions = []
    for kind, n in Counter(kinds).items():
        saved = saved_tokens(n, prefix)
        decisions.append(WarmupDecision(kind=kind, questions=n, prefix_tokens=prefix, saved_tokens=saved,
                                        warm=n >= min_questions and saved >= min_saved_tokens))
    return decisions


def image_size(path: str) -> Tuple[int, int]:
    """Width and height from the file header; the pixels are not decoded."""
    from PIL import Image
    with Image.open(path) as image:
        return image.size


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
        self.logger.info(f"downloading {WEIGHTS_ID}@{WEIGHTS_REVISION[:8]}")
        # HF_TOKEN reads the repository while it is private.
        model_dir = await asyncio.to_thread(snapshot_download, WEIGHTS_ID, revision=WEIGHTS_REVISION, token=os.environ.get("HF_TOKEN"))
        self.logger.info(f"downloaded in {time.monotonic() - started:.1f}s")

        self.image_grid = ImageGrid.from_preprocessor_config(os.path.join(model_dir, "preprocessor_config.json"))

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        self.base_url = f"http://127.0.0.1:{port}"
        self.server_log: deque = deque(maxlen=SERVER_LOG_TAIL)
        self.server = await asyncio.create_subprocess_exec(
            sys.executable, f"{model_dir}/serve_decide.py",
            "--model", model_dir,
            "--served-model-name", MODEL_ID,
            "--logprobs-mode", "processed_logprobs",
            "--max-model-len", str(MAX_MODEL_LEN),
            "--enable-prefix-caching", "--mamba-cache-mode", "align", "--prefix-match-unit", str(PREFIX_MATCH_UNIT),
            "--limit-mm-per-prompt", json.dumps({"image": MAX_IMAGES}),
            "--max-num-seqs", str(MAX_NUM_SEQS),
            "--trust-request-chat-template",
            "--host", "127.0.0.1", "--port", str(port),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env={**os.environ, "JEV_DECIDE_LORA": MODEL_ID, "JEV_DECIDE_ADAPTER_DIR": f"{model_dir}/adapter_vllm"},
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

    async def _warm(self, kind: str, image_urls: List[str], text: str) -> int:
        """Prefill the shared prefix of `kind` questions. Returns its prompt tokens, 0 on failure."""
        path, body = warm_request(kind, image_urls, text)
        try:
            response = await self.client.post(path, json=body)
            if response.status_code == 200:
                return int((response.json().get("usage") or {}).get("prompt_tokens") or 0)
            self.logger.warning(f"warm-up for {kind} questions failed ({response.status_code}): {response.text[:300]}")
        except httpx.HTTPError as e:
            self.logger.warning(f"warm-up for {kind} questions failed: {e}")
        return 0

    async def _answer(self, image_urls: List[str], image_sizes: List[Tuple[int, int]], text: str,
                      requests: List[Tuple[str, Dict[str, Any], int]], warm: Optional[bool] = None) -> List[Dict[str, Any]]:
        """Every question against the state. A kind whose questions share enough prefix first gets a
        warm-up request (see decide_warmups), so vLLM checkpoints the shared prefix and each question
        prefills only its own tokens. The questions' inputs are unchanged; a failed warm-up only costs
        the speedup. `warm` None decides per kind; True or False forces it for every kind."""
        state = await asyncio.to_thread(encode_state, image_urls, text)
        decisions = decide_warmups(image_sizes, text, [body["kind"] for _, body, _ in requests], self.image_grid)
        kinds = [d.kind for d in decisions if (d.warm if warm is None else warm)]
        if kinds:
            await asyncio.gather(*(self._warm(k, image_urls, text) for k in kinds))
        return await asyncio.gather(*(self._decide(qid, state, body) for qid, body, _ in requests))

    async def run(self, input_data: AppInput) -> AppOutput:
        """Evaluate the state and images against every question."""
        if self.server.returncode is not None:
            raise RuntimeError(f"the vLLM server exited with code {self.server.returncode}:\n" + "\n".join(self.server_log))
        for image in input_data.images:
            if not image.exists():
                raise RuntimeError(f"image does not exist at path: {image.path}")

        image_urls = await asyncio.to_thread(lambda: [image_part(image)["image"] for image in input_data.images])
        image_sizes = await asyncio.to_thread(lambda: [image_size(image.path) for image in input_data.images])
        requests = build_requests(input_data)
        started = time.monotonic()
        results = await self._answer(image_urls, image_sizes, as_text(input_data.state), requests)
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

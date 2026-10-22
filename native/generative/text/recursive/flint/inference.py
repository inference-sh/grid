"""
Recursive Flint: a 40B-A2.5B mixture-of-experts coding-agent model, run on our own GPUs.

Serving follows the model card. The repository's SGLang kit (sglang/) is the only supported
runtime: SGLang 0.5.16 plus the `flint_sglang` model package, FlashInfer 0.6.16.post1, CUTLASS
DSL 4.6.2 and, on Blackwell, a patch to SGLang's FA4 attention kernel. The author ships it as a
Docker image; here requirements.txt installs SGLang 0.5.16 and its CUDA 13 stack from PyPI, and
setup() runs the kit's own install.sh against this venv once per app version. install.sh
force-reinstalls the pinned FlashInfer / CUTLASS wheels with --no-deps, which a lock file cannot
express, applies the FA4 patch and installs the wheel shipped in the repository.

setup() then starts the kit's serve script for this GPU (serve_h200.sh or serve_b300.sh, the
configurations the model was evaluated with) as a child process on localhost and waits for
/health_generate. run() streams /v1/chat/completions.

Hardware: one GPU with at least 141 GB (H200, B200, B300). The BF16 weights are 79.6 GB and the
scripts give the rest to the KV cache, so the app takes the whole card. The kit needs an NVIDIA
driver of at least 580 (CUDA 13); older datacenter drivers work only with CUDA 13 forward
compatibility libraries, which the worker image does not ship today.

Request rules from the kit's README, applied here:
- Text only. Message content goes out as a string; images and files are refused.
- Reasoning arrives as `reasoning_content` and is sent back on assistant messages.
- Sampling defaults are the card's: temperature 1.0, top_p 0.95, top_k 20.
- Reasoning counts toward max_tokens. The card asks for at least 32768 on hard problems.
- A streamed tool call whose last argument is a JSON object can lose its final `}`; the final
  output repairs that.

Model: https://huggingface.co/RecursiveSI/flint (Apache 2.0)
"""

import asyncio
import json
import logging
import os
import re
import shutil
import socket
import sys
import time
from collections import deque
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import httpx
from inferencesh import BaseApp, OutputMeta
from inferencesh.models.llm import (
    LLMDelta,
    LLMInput,
    LLMOutput,
    LLMUsage,
    build_openai_messages,
    build_tools,
    openai_response_format,
    openai_tool_choice,
)
from inferencesh.openai import OpenAIChatMixin
from pydantic import Field

MODEL_ID = "RecursiveSI/flint"
# setup() executes the repository's install.sh, serve scripts and flint_sglang wheel.
# Review the diff before moving this.
REVISION = "2a247be0c0aec895ccb0ac7ca92229ec6df37288"
SERVED_MODEL_NAME = "flint"
CONTEXT_LENGTH = 262144
# The card's recommended output budget for hard problems; reasoning counts toward it.
DEFAULT_MAX_TOKENS = 32768
MIN_VRAM_GB = 140
# Driver 580 is the first with CUDA 13. Below it the kit needs forward-compatibility libraries.
MIN_DRIVER_MAJOR = 580
SERVER_LOG_TAIL = 60
KIT_MARKER = "flint-kit-installed"

# GPU name substring -> serve script in sglang/. Blackwell uses FA4 + trtllm MoE, Hopper FA3 + Triton.
SERVE_SCRIPTS = [
    ("B300", "serve_b300.sh"),
    ("B200", "serve_b300.sh"),
    ("GB300", "serve_b300.sh"),
    ("GB200", "serve_b300.sh"),
    ("H200", "serve_h200.sh"),
]


# ── Input and output ─────────────────────────────────────────────────────────

class AppInput(LLMInput):
    """Recursive Flint: text input, tools, reasoning."""
    system_prompt: str = Field(default="", description="System prompt. Flint has no default system prompt.")
    context_size: int = Field(default=CONTEXT_LENGTH, description="The context size for the model. Prompt and output share it.")
    temperature: float = Field(default=1.0, ge=0.0, le=1.0, description="Sampling temperature. The card's setting is 1.0; greedy decoding is not validated.")
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=20, description="Top-k sampling. The card's setting is 20.")
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, ge=1, le=CONTEXT_LENGTH, description="Output budget, reasoning included. Use at least 32768 for hard problems.")


class AppOutput(LLMOutput):
    """Response text, reasoning, tool calls and token usage."""
    pass


# ── Request building ─────────────────────────────────────────────────────────

def _has_media(input_data: AppInput) -> bool:
    if input_data.images or input_data.files or getattr(input_data, "attachments", None):
        return True
    return any(getattr(m, "images", None) or getattr(m, "files", None) for m in input_data.context)


def _flatten_content(content: Any) -> Any:
    """Flint's template takes string content only: join the text parts of a content array."""
    if not isinstance(content, list):
        return content
    texts = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            texts.append(part.get("text") or "")
        elif isinstance(part, str):
            texts.append(part)
        else:
            raise ValueError("Flint accepts text only. Remove images and files from the conversation.")
    return "\n".join(texts)


def build_request_body(input_data: AppInput) -> Dict[str, Any]:
    if _has_media(input_data):
        raise ValueError("Flint accepts text only. Remove images and files from the conversation.")

    messages = build_openai_messages(input_data, reasoning_field="reasoning_content")
    for message in messages:
        if "content" in message:
            message["content"] = _flatten_content(message["content"])

    body: Dict[str, Any] = {
        "model": SERVED_MODEL_NAME,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": input_data.max_tokens,
        "temperature": input_data.temperature,
        "top_p": input_data.top_p,
    }
    if input_data.top_k is not None and input_data.top_k > 0:
        body["top_k"] = input_data.top_k
    if input_data.min_p is not None and input_data.min_p > 0:
        body["min_p"] = input_data.min_p
    if input_data.frequency_penalty is not None:
        body["frequency_penalty"] = input_data.frequency_penalty
    if input_data.presence_penalty is not None:
        body["presence_penalty"] = input_data.presence_penalty
    if input_data.repetition_penalty is not None:
        body["repetition_penalty"] = input_data.repetition_penalty
    if input_data.seed is not None:
        body["seed"] = input_data.seed
    if input_data.stop:
        body["stop"] = input_data.stop

    tools = build_tools(input_data.tools) if input_data.tools else None
    if tools:
        body["tools"] = tools
        body["tool_choice"] = openai_tool_choice(input_data.tool_choice)

    response_format = openai_response_format(input_data.response_format)
    if response_format is not None:
        body["response_format"] = response_format
    # reasoning_effort is not sent: Flint decides per turn whether to reason, with no switch or budget.
    return body


# ── Stream handling ──────────────────────────────────────────────────────────

def new_state() -> Dict[str, Any]:
    return {
        "response": "",
        "reasoning": "",
        "tool_calls": [],
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
        "finish_reason": None,
        "started_at": time.monotonic(),
        "first_token_at": None,
    }


def _tool_call_delta(tc: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """Fold one tool_call delta into state; return it with a resolved index and no nulls."""
    calls = state["tool_calls"]
    idx = tc.get("index")
    if idx is None:
        idx = len(calls) if tc.get("id") or not calls else len(calls) - 1
    while len(calls) <= idx:
        calls.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
    current = calls[idx]
    if tc.get("id"):
        current["id"] = tc["id"]
    fn = tc.get("function") or {}
    if fn.get("name"):
        current["function"]["name"] = fn["name"]
    args = fn.get("arguments")
    if args and not isinstance(args, str):
        args = json.dumps(args)
    if args:
        current["function"]["arguments"] += args

    out: Dict[str, Any] = {"index": idx}
    if tc.get("id"):
        out["id"] = tc["id"]
        out["type"] = "function"
    out_fn = {k: v for k, v in (("name", fn.get("name")), ("arguments", args)) if v}
    if out_fn:
        out["function"] = out_fn
    return out


def handle_chunk(data: Dict[str, Any], state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Fold one SSE chunk into state. Returns an LLMDelta-shaped dict, or None."""
    error = data.get("error")
    if error:
        err = error if isinstance(error, dict) else {"message": str(error)}
        raise RuntimeError(f"Flint server error: {err.get('message', 'unknown error')}")

    usage = data.get("usage")
    if usage:
        state["input_tokens"] = usage.get("prompt_tokens") or 0
        state["output_tokens"] = usage.get("completion_tokens") or 0
        state["cached_tokens"] = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        state["reasoning_tokens"] = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0

    choices = data.get("choices") or []
    if not choices:
        return None
    choice = choices[0]
    if choice.get("finish_reason"):
        state["finish_reason"] = choice["finish_reason"]

    delta = choice.get("delta") or {}
    out: Dict[str, Any] = {}
    reasoning = delta.get("reasoning_content")
    if reasoning:
        state["reasoning"] += reasoning
        out["reasoning"] = reasoning
    content = delta.get("content")
    if content:
        state["response"] += content
        out["response"] = content
    tool_calls = delta.get("tool_calls")
    if tool_calls:
        out["tool_calls"] = [_tool_call_delta(tc, state) for tc in tool_calls]

    if out and state["first_token_at"] is None:
        state["first_token_at"] = time.monotonic()
    return out or None


def repair_arguments(arguments: str) -> str:
    """Add back closing braces a streamed call can lose (kit README: only the response's last
    call is repaired by SGLang). Returns the input unchanged when it parses or cannot be fixed."""
    if not arguments:
        return arguments
    try:
        json.loads(arguments)
        return arguments
    except json.JSONDecodeError:
        pass
    fixed = arguments
    for _ in range(4):
        fixed += "}"
        try:
            json.loads(fixed)
            return fixed
        except json.JSONDecodeError:
            continue
    return arguments


def build_output(state: Dict[str, Any], final: bool = False) -> Dict[str, Any]:
    out: Dict[str, Any] = {"response": state["response"]}
    if state["reasoning"]:
        out["reasoning"] = state["reasoning"]
    if state["tool_calls"]:
        out["tool_calls"] = state["tool_calls"]
    if not final:
        return out

    for call in state["tool_calls"]:
        call["function"]["arguments"] = repair_arguments(call["function"]["arguments"])
    elapsed = max(time.monotonic() - state["started_at"], 1e-6)
    ttft = (state["first_token_at"] - state["started_at"]) if state["first_token_at"] else 0.0
    gen_time = max(elapsed - ttft, 1e-6)
    out["usage"] = LLMUsage(
        stop_reason=state["finish_reason"] or "",
        time_to_first_token=round(ttft, 3),
        tokens_per_second=round(state["output_tokens"] / gen_time, 2) if state["output_tokens"] else 0.0,
        prompt_tokens=state["input_tokens"],
        cached_tokens=state["cached_tokens"],
        # completion_tokens already includes reasoning tokens (OpenAI convention).
        completion_tokens=state["output_tokens"],
        total_tokens=state["input_tokens"] + state["output_tokens"],
        reasoning_tokens=state["reasoning_tokens"],
    )
    out["output_meta"] = OutputMeta.from_llm_usage(out["usage"])
    return out


# ── Worker checks ────────────────────────────────────────────────────────────

def parse_gpus(nvidia_smi_csv: str) -> List[Dict[str, Any]]:
    """Rows of `nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits`."""
    gpus = []
    for line in nvidia_smi_csv.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        name, memory_mib, driver = parts[0], parts[1], parts[2]
        gpus.append({"name": name, "memory_gb": float(memory_mib) / 1024 if memory_mib.replace(".", "").isdigit() else 0.0, "driver": driver})
    return gpus


def pick_serve_script(gpu: Dict[str, Any]) -> str:
    name = gpu["name"].upper()
    for key, script in SERVE_SCRIPTS:
        if key in name:
            return script
    raise RuntimeError(
        f"Flint runs on one H200, B200 or B300 (141 GB or more per GPU); this worker has {gpu['name']} "
        f"with {gpu['memory_gb']:.0f} GB. 80 GB GPUs cannot hold the BF16 weights."
    )


FA4_KERNEL = os.path.join("jit_kernel", "flash_attn", "cute", "flash_fwd_sm100.py")


def unlink_patched_kernel() -> None:
    """Replace the FA4 kernel file install.sh patches with a regular copy. Worker venvs link
    package files into the shared uv cache (UV_LINK_MODE=symlink), and `patch` refuses to edit a
    symlink. Copying keeps the cache's file untouched."""
    import importlib.util

    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("sglang is not installed in this environment")
    path = os.path.join(list(spec.submodule_search_locations)[0], FA4_KERNEL)
    if os.path.islink(path):
        target = os.path.realpath(path)
        tmp = path + ".copy"
        shutil.copyfile(target, tmp)
        os.replace(tmp, path)


def driver_major(driver: str) -> int:
    match = re.match(r"(\d+)", driver or "")
    return int(match.group(1)) if match else 0


# ── App ──────────────────────────────────────────────────────────────────────

class App(OpenAIChatMixin, BaseApp):
    async def setup(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        started = time.monotonic()

        gpu = await self._check_gpu()
        script = pick_serve_script(gpu)

        from huggingface_hub import snapshot_download

        self.logger.info(f"downloading {MODEL_ID}@{REVISION[:8]} (79.6 GB)")
        model_dir = await asyncio.to_thread(snapshot_download, MODEL_ID, revision=REVISION, token=os.environ.get("HF_TOKEN"))
        self.logger.info(f"downloaded in {time.monotonic() - started:.1f}s")

        self.server_log: deque = deque(maxlen=SERVER_LOG_TAIL)
        env = self._server_env()
        await self._install_kit(model_dir, env)
        await self._start_server(model_dir, script, env)
        self.logger.info(f"Flint ready on {gpu['name']} with {script} in {time.monotonic() - started:.1f}s")

    async def _check_gpu(self) -> Dict[str, Any]:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        out, _ = await proc.communicate()
        gpus = parse_gpus(out.decode(errors="replace"))
        if proc.returncode != 0 or not gpus:
            raise RuntimeError(f"nvidia-smi failed: {out.decode(errors='replace')[:500]}")
        gpu = gpus[0]
        self.logger.info(f"GPU {gpu['name']}, {gpu['memory_gb']:.0f} GB, driver {gpu['driver']}")
        if driver_major(gpu["driver"]) < MIN_DRIVER_MAJOR:
            self.logger.warning(
                f"driver {gpu['driver']} is older than {MIN_DRIVER_MAJOR} (CUDA 13). The kit's CUDA 13 wheels "
                "need CUDA forward-compatibility libraries on this worker, or the server will fail to start."
            )
        return gpu

    def _server_env(self) -> Dict[str, str]:
        venv_bin = os.path.dirname(sys.executable)
        env = {**os.environ, "PATH": f"{venv_bin}:{os.environ.get('PATH', '')}"}
        # Keep compiled kernels in the app's caches so a restart does not recompile them.
        cache_root = os.environ.get("TRITON_CACHE_DIR") or os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache")), "..", "triton")
        env["FLINT_TRITON_CACHE_BASE"] = cache_root
        env.setdefault("FLASHINFER_WORKSPACE_BASE", os.path.join(cache_root, "flashinfer"))
        env["FLASHINFER_USE_CUDA_NORM"] = "1"
        env["SGLANG_EXTERNAL_MODEL_PACKAGE"] = "flint_sglang"
        return env

    async def _install_kit(self, model_dir: str, env: Dict[str, str]) -> None:
        """Run the kit's install.sh against this venv once per app version (the venv persists)."""
        marker = os.path.join(sys.prefix, f"{KIT_MARKER}-{REVISION[:12]}")
        if os.path.exists(marker):
            self.logger.info("Flint SGLang kit already installed")
            return
        if shutil.which("patch", path=env["PATH"]) is None:
            raise RuntimeError("the `patch` command is missing; packages.txt must list it")
        kit_dir = os.path.join(model_dir, "sglang")
        unlink_patched_kernel()
        started = time.monotonic()
        self.logger.info("installing the Flint SGLang kit (install.sh)")
        proc = await asyncio.create_subprocess_exec(
            "bash", os.path.join(kit_dir, "install.sh"), kit_dir,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, env=env,
        )
        tail: deque = deque(maxlen=SERVER_LOG_TAIL)
        async for raw in proc.stdout:
            line = raw.decode(errors="replace").rstrip()
            if line:
                tail.append(line)
                self.logger.info(f"[install] {line}")
        await proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"Flint install.sh exited with code {proc.returncode}:\n" + "\n".join(tail))
        with open(marker, "w") as f:
            f.write(REVISION)
        self.logger.info(f"kit installed in {time.monotonic() - started:.1f}s")

    async def _start_server(self, model_dir: str, script: str, env: Dict[str, str]) -> None:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        env = {**env, "MODEL": model_dir, "PORT": str(port)}
        self.server = await asyncio.create_subprocess_exec(
            # Arguments after the script are appended to launch_server; a repeated flag overrides.
            # --enable-cache-report fills usage.prompt_tokens_details.cached_tokens (otherwise always 0).
            "bash", os.path.join(model_dir, "sglang", script), "--host", "127.0.0.1", "--enable-cache-report",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, env=env,
        )
        self.server_log_task = asyncio.create_task(self._forward_server_log())
        self.client = httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=httpx.Timeout(None, connect=10.0))

        # No deadline here: the platform owns the setup timeout. Stop only if the server exits.
        while True:
            if self.server.returncode is not None:
                raise RuntimeError(f"the SGLang server exited with code {self.server.returncode} during startup:\n" + "\n".join(self.server_log))
            try:
                if (await self.client.get("/health_generate", timeout=60.0)).status_code == 200:
                    return
            except httpx.TransportError:
                pass
            await asyncio.sleep(5)

    async def _forward_server_log(self):
        async for raw in self.server.stdout:
            line = raw.decode(errors="replace").rstrip()
            if line:
                self.server_log.append(line)
                self.logger.info(f"[sglang] {line}")

    async def run(self, input_data: AppInput) -> AsyncGenerator[Union[LLMDelta, AppOutput], None]:
        if self.server.returncode is not None:
            raise RuntimeError(f"the SGLang server exited with code {self.server.returncode}:\n" + "\n".join(self.server_log))

        body = build_request_body(input_data)
        state = new_state()
        chunks = 0
        self.logger.info(
            f"request messages={len(body['messages'])} tools={len(body.get('tools') or [])} max_tokens={body['max_tokens']}"
        )
        async with self.client.stream("POST", "/v1/chat/completions", json=body) as resp:
            if resp.status_code != 200:
                raw = (await resp.aread()).decode(errors="replace")
                try:
                    message = (json.loads(raw).get("error") or {}).get("message") or raw
                except (ValueError, AttributeError):
                    message = raw
                raise RuntimeError(f"Flint server returned {resp.status_code}: {message[:2000]}")
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if not data_str or data_str == "[DONE]":
                    continue
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    self.logger.warning(f"bad SSE JSON: {data_str[:200]}")
                    continue
                chunks += 1
                delta = handle_chunk(data, state)
                if delta:
                    yield LLMDelta(**delta)

        final = build_output(state, final=True)
        self.logger.info(
            f"answered by {MODEL_ID}: chunks={chunks} finish={state['finish_reason']} in={state['input_tokens']} "
            f"cached={state['cached_tokens']} out={state['output_tokens']} reasoning={state['reasoning_tokens']} "
            f"tool_calls={len(state['tool_calls'])} ttft={final['usage'].time_to_first_token}s "
            f"tps={final['usage'].tokens_per_second}"
        )
        yield AppOutput(**final)

    async def unload(self):
        if hasattr(self, "client"):
            await self.client.aclose()
        if hasattr(self, "server") and self.server.returncode is None:
            self.server.terminate()
            try:
                await asyncio.wait_for(self.server.wait(), timeout=30)
            except asyncio.TimeoutError:
                self.server.kill()
        if hasattr(self, "server_log_task"):
            self.server_log_task.cancel()

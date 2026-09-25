"""Melious Chat Completions streaming helper — raw httpx SSE, no SDK.

Shared by the melious/* chat apps (symlinked into each app directory).
Melious (https://api.melious.ai) serves open-weight models behind an
OpenAI-compatible /v1/chat/completions. Differences from OpenAI handled here:

- Reasoning text streams as `delta.reasoning_content`.
- `reasoning_effort` takes low / medium / high, the same set as LLMInput. The
  platform default "none" sends nothing, so the model runs at its own default.
- Usage arrives on a final chunk with empty choices; cached and reasoning
  tokens are in prompt_tokens_details / completion_tokens_details.
- qwen3.6-27b's thinking is hidden upstream: reasoning_content stays null,
  reasoning_tokens is 0 though completion_tokens includes the thinking, and
  the content starts with the blank lines that followed </think>. Content is
  passed through unchanged.
- Streamed responses carry no `billing_cost` / `environment_impact`.
- Follow-up tool_call chunks carry `"name": null`; nulls are dropped.
- A stream can open and then go silent (seen once on deepseek-v4.1-flash).
  Keep-alives arrive every ~10s, so 60s of silence is treated as a dead
  connection and the request is sent again if nothing was emitted yet.
- File input is not documented by the API, so it is rejected. Images are sent
  as data URIs to models that accept them and rejected by the others.
"""

import asyncio
import json
import os
import time
from enum import Enum
from typing import Any, AsyncGenerator, Dict, Optional

import httpx

from inferencesh import OutputMeta, TextMeta
from inferencesh.models.llm import (
    LLMUsage,
    build_openai_messages,
    build_tools,
    openai_response_format,
    openai_tool_choice,
)

MELIOUS_BASE_URL = "https://api.melious.ai/v1"

# Attempts per request. Only failures before any output is emitted are retried.
MAX_ATTEMPTS = 3
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}

# Melious sends an SSE keep-alive comment about every 10s while a model thinks
# without streaming (measured on qwen3.6-27b), so a minute with no bytes at all
# means the connection is dead, not that the model is slow.
STREAM_SILENCE_TIMEOUT = 60.0


def get_api_key() -> str:
    key = os.environ.get("MELIOUS_KEY")
    if not key:
        raise RuntimeError(
            "MELIOUS_KEY is not set. A secret whose record exists but holds an empty value is "
            "not injected at all: check that `belt secrets get MELIOUS_KEY --json` reports a "
            "non-empty masked_value, and re-set it if it does not."
        )
    return key.strip()


# ---------------------------------------------------------------------------
# Request building
# ---------------------------------------------------------------------------

def _has_images(input_data) -> bool:
    return bool(input_data.images) or any(getattr(m, "images", None) for m in input_data.context)


def _has_files(input_data) -> bool:
    return (
        bool(input_data.files)
        or bool(getattr(input_data, "attachments", None))
        or any(getattr(m, "files", None) for m in input_data.context)
    )


def _effort(input_data) -> Optional[str]:
    """LLMInput.reasoning_effort (none/low/medium/high) -> Melious reasoning_effort, or None to send nothing."""
    effort = input_data.reasoning_effort
    if isinstance(effort, Enum):
        effort = effort.value
    return effort if effort and effort != "none" else None


def build_request_body(input_data, model: str, *, vision: bool) -> Dict[str, Any]:
    if _has_files(input_data):
        raise ValueError("File input is not supported by the Melious apps. Paste the text into the message instead.")
    if _has_images(input_data) and not vision:
        raise ValueError(f"{model} does not accept image input.")

    messages = build_openai_messages(input_data, file_mode="base64", image_mode="base64")
    body: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": input_data.max_tokens,
    }

    if input_data.temperature is not None:
        body["temperature"] = input_data.temperature
    if input_data.top_p is not None:
        body["top_p"] = input_data.top_p
    if input_data.top_k is not None and input_data.top_k >= 0:
        body["top_k"] = input_data.top_k
    if input_data.min_p is not None and input_data.min_p > 0:
        body["min_p"] = input_data.min_p
    if input_data.frequency_penalty is not None:
        body["frequency_penalty"] = input_data.frequency_penalty
    if input_data.presence_penalty is not None:
        body["presence_penalty"] = input_data.presence_penalty
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

    effort = _effort(input_data)
    if effort:
        body["reasoning_effort"] = effort

    return body


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

def _format_error(status: int, body: Any, headers: Dict[str, str]) -> str:
    request_id = headers.get("x-request-id", "")
    err = body.get("error", {}) if isinstance(body, dict) else {}
    if isinstance(err, str):
        err = {"message": err}
    msg = err.get("message") or (body if isinstance(body, str) else f"HTTP {status}")
    detail = " ".join(str(p) for p in (status, err.get("code"), err.get("type")) if p)
    text = f"Melious API error ({detail}): {msg}"
    if err.get("details"):
        text += f" {json.dumps(err['details'])}"
    if request_id:
        text += f" [req:{request_id}]"
    return text


# ---------------------------------------------------------------------------
# SSE chunk handling
# ---------------------------------------------------------------------------

def _create_initial_state() -> Dict[str, Any]:
    return {
        "response": "",
        "reasoning": "",
        "tool_calls": [],
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
        "finish_reason": None,
        "completion_id": None,
        "started_at": time.monotonic(),
        "first_token_at": None,
    }


def _tool_call_delta(tc: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """Fold one tool_call delta into state; return it with a resolved index."""
    calls = state["tool_calls"]
    idx = tc.get("index")
    if idx is None and tc.get("id"):
        idx = next((i for i, c in enumerate(calls) if c["id"] == tc["id"]), None)
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

    # Melious sends null for fields a chunk does not carry; LLMDelta rejects None.
    out: Dict[str, Any] = {"index": idx}
    if tc.get("id"):
        out["id"] = tc["id"]
        out["type"] = "function"
    out_fn = {k: v for k, v in (("name", fn.get("name")), ("arguments", args)) if v}
    if out_fn:
        out["function"] = out_fn
    return out


def _handle_chunk(data: Dict[str, Any], state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Fold one chunk into state. Returns an LLMDelta-shaped dict, or None."""
    error = data.get("error")
    if error:
        err = error if isinstance(error, dict) else {"message": str(error)}
        code = err.get("code") or err.get("type")
        raise RuntimeError(f"Melious error{f' ({code})' if code else ''}: {err.get('message', 'unknown error')}")

    if data.get("id") and not state["completion_id"]:
        state["completion_id"] = data["id"]

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
        if choice["finish_reason"] == "error":
            raise RuntimeError("Melious stream terminated with finish_reason=error")

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


# ---------------------------------------------------------------------------
# Output building
# ---------------------------------------------------------------------------

def _build_output(state: Dict[str, Any], final: bool = False) -> Dict[str, Any]:
    out: Dict[str, Any] = {"response": state["response"]}
    if state["reasoning"]:
        out["reasoning"] = state["reasoning"]
    if state["tool_calls"]:
        out["tool_calls"] = state["tool_calls"]
    if not final:
        return out

    # completion_tokens already includes reasoning tokens (OpenAI convention).
    inputs = []
    if state["input_tokens"]:
        inputs.append(TextMeta(tokens=state["input_tokens"], extra={
            "cache_read_tokens": state["cached_tokens"],
        }))
    outputs = [TextMeta(tokens=state["output_tokens"], extra={
        "reasoning_tokens": state["reasoning_tokens"],
    })]
    out["output_meta"] = OutputMeta(inputs=inputs, outputs=outputs)

    elapsed = max(time.monotonic() - state["started_at"], 1e-6)
    ttft = (state["first_token_at"] - state["started_at"]) if state["first_token_at"] else 0.0
    gen_time = max(elapsed - ttft, 1e-6)
    out["usage"] = LLMUsage(
        stop_reason=state["finish_reason"] or "",
        time_to_first_token=round(ttft, 3),
        tokens_per_second=round(state["output_tokens"] / gen_time, 2) if state["output_tokens"] else 0.0,
        prompt_tokens=state["input_tokens"],
        completion_tokens=state["output_tokens"],
        total_tokens=state["input_tokens"] + state["output_tokens"],
        reasoning_tokens=state["reasoning_tokens"],
    )
    return out


# ---------------------------------------------------------------------------
# Main streaming function
# ---------------------------------------------------------------------------

class _Retryable(Exception):
    """A failure before any output was emitted; the request can be sent again."""


async def _open_stream(http: httpx.AsyncClient, body: Dict[str, Any], headers: Dict[str, str]) -> httpx.Response:
    """POST the request and return the open 200 stream.

    Raises _Retryable for connection failures and transient statuses, and
    RuntimeError for everything else.
    """
    try:
        req = http.build_request("POST", f"{MELIOUS_BASE_URL}/chat/completions", json=body, headers=headers)
        resp = await http.send(req, stream=True)
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.WriteTimeout, httpx.PoolTimeout) as e:
        raise _Retryable(f"connection failed ({type(e).__name__}: {e})")

    if resp.status_code == 200:
        return resp

    raw = await resp.aread()
    await resp.aclose()
    try:
        err_body = json.loads(raw)
    except json.JSONDecodeError:
        err_body = raw.decode("utf-8", "replace")[:500]
    message = _format_error(resp.status_code, err_body, dict(resp.headers))
    if resp.status_code in RETRYABLE_STATUS:
        raise _Retryable(message)
    raise RuntimeError(message)


async def stream_completion(
    input_data,
    model: str,
    *,
    vision: bool = False,
    with_deltas: bool = False,
) -> AsyncGenerator[Any, None]:
    """Stream a Melious chat completion.

    with_deltas=False: yields accumulated output dicts (last one carries usage + output_meta).
    with_deltas=True:  yields (output_dict, delta_dict | None) tuples; delta_dict has
                       LLMDelta-shaped keys (response, reasoning, tool_calls).

    A request that fails before any output was emitted (connection error,
    transient status, silent or dropped stream) is sent again, up to
    MAX_ATTEMPTS. Once output has been emitted a failure is raised: resending
    would duplicate what the caller already has.
    """
    body = build_request_body(input_data, model, vision=vision)
    headers = {
        "Authorization": f"Bearer {get_api_key()}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    timeout = httpx.Timeout(connect=15.0, read=STREAM_SILENCE_TIMEOUT, write=30.0, pool=15.0)

    async with httpx.AsyncClient(timeout=timeout) as http:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            print(
                f"Calling Melious model={model} attempt={attempt} messages={len(body['messages'])}"
                f" tools={len(body.get('tools') or [])} reasoning_effort={body.get('reasoning_effort')}"
                f" max_tokens={body['max_tokens']}"
            )
            state = _create_initial_state()
            chunks = keepalives = 0
            max_gap = 0.0
            emitted = False
            try:
                resp = await _open_stream(http, body, headers)
                print(f"Stream opened model={model} req={resp.headers.get('x-request-id') or 'unknown'}")
                last_line_at = time.monotonic()
                try:
                    async for line in resp.aiter_lines():
                        now = time.monotonic()
                        max_gap = max(max_gap, now - last_line_at)
                        last_line_at = now
                        if line.startswith(":"):
                            keepalives += 1
                            continue
                        if not line.startswith("data:"):
                            continue
                        data_str = line[5:].strip()
                        if not data_str or data_str == "[DONE]":
                            continue
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            print(f"  Bad SSE JSON: {data_str[:200]}")
                            continue

                        chunks += 1
                        delta = _handle_chunk(data, state)
                        if delta is None:
                            continue
                        emitted = True
                        output = _build_output(state)
                        yield (output, delta) if with_deltas else output
                except httpx.ReadTimeout:
                    failure = f"no data for {STREAM_SILENCE_TIMEOUT:.0f}s after {chunks} chunks"
                    if emitted:
                        raise RuntimeError(f"Melious stream stalled: {failure}")
                    raise _Retryable(failure)
                except (httpx.RemoteProtocolError, httpx.ReadError) as e:
                    failure = f"stream dropped after {chunks} chunks ({type(e).__name__}: {e})"
                    if emitted:
                        raise RuntimeError(f"Melious {failure}")
                    raise _Retryable(failure)
                finally:
                    await resp.aclose()
            except _Retryable as e:
                if attempt == MAX_ATTEMPTS:
                    raise RuntimeError(f"Melious request failed after {attempt} attempts: {e}")
                backoff = 2.0 ** attempt
                print(f"  {e}; retrying in {backoff:.0f}s")
                await asyncio.sleep(backoff)
                continue
            break

    print(
        f"Stream done model={model} id={state['completion_id'] or '?'} chunks={chunks}"
        f" finish={state['finish_reason']} in={state['input_tokens']} cached={state['cached_tokens']}"
        f" out={state['output_tokens']} reasoning={state['reasoning_tokens']}"
        f" tool_calls={len(state['tool_calls'])} keepalives={keepalives} max_gap={max_gap:.1f}s"
    )

    final = _build_output(state, final=True)
    yield (final, None) if with_deltas else final

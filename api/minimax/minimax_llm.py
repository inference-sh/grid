"""MiniMax LLM streaming helper — raw httpx SSE against OpenAI-compatible endpoint."""

from __future__ import annotations

import asyncio
import json
import time
from typing import List, Optional, Dict, Any, AsyncGenerator, Tuple

import httpx

from inferencesh import OutputMeta, TextMeta
from inferencesh.llm_types_gen import ResponseFormatType, ToolChoiceMode
from inferencesh.models.llm import build_openai_messages, build_tools

MINIMAX_BASE_URL = "https://api.minimax.io/v1"
STREAM_SILENCE_TIMEOUT = 60


async def stream_completion(
    api_key: str, input_data, model: str, *, with_deltas: bool = False,
) -> AsyncGenerator[Any, None]:
    """Stream a completion from MiniMax.

    with_deltas=False (default): yields accumulated output dicts.
    with_deltas=True: yields (output_dict, delta_dict | None) tuples, where
    delta_dict has LLMDelta-shaped keys (response, reasoning, tool_calls).
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    body = _build_request_body(input_data, model)
    print(f"Calling MiniMax model={model} max_tokens={body.get('max_tokens')}")

    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10, read=STREAM_SILENCE_TIMEOUT, write=10, pool=10)) as http:
        req = http.build_request("POST", f"{MINIMAX_BASE_URL}/chat/completions", json=body, headers=headers)
        resp = await asyncio.wait_for(http.send(req, stream=True), timeout=15.0)

        if resp.status_code != 200:
            error_text = (await resp.aread()).decode()
            await resp.aclose()
            try:
                error_msg = json.loads(error_text).get("error", {}).get("message", error_text[:300])
            except Exception:
                error_msg = error_text[:300]
            raise RuntimeError(f"MiniMax API error ({resp.status_code}): {error_msg}")

        state = _create_initial_state()
        chunks_received = 0

        try:
            async for line in resp.aiter_lines():
                if line.startswith(":"):
                    continue
                if not line.startswith("data: "):
                    continue

                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    print(
                        f"Stream done model={model} chunks={chunks_received}"
                        f" in={state.get('input_tokens', 0)} out={state.get('output_tokens', 0)}"
                        f" reasoning={state.get('reasoning_tokens', 0)}"
                    )
                    break

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                chunks_received += 1
                _, chunk_delta = _parse_sse_chunk(data, state)
                output = _build_output(state)
                yield (output, chunk_delta) if with_deltas else output

        except httpx.ReadTimeout:
            raise RuntimeError(f"Stream timed out — no data for {STREAM_SILENCE_TIMEOUT}s after {chunks_received} chunks")
        finally:
            await resp.aclose()


def _build_request_body(input_data, model: str) -> Dict[str, Any]:
    messages = build_openai_messages(input_data, file_mode="url", image_mode="url")
    tools = build_tools(input_data.tools) if input_data.tools else None

    body: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": getattr(input_data, "max_tokens", 32768),
    }

    if input_data.temperature is not None:
        body["temperature"] = input_data.temperature
    if input_data.top_p is not None:
        body["top_p"] = input_data.top_p
    if input_data.top_k is not None and input_data.top_k >= 0:
        body["top_k"] = input_data.top_k
    if input_data.frequency_penalty is not None:
        body["frequency_penalty"] = input_data.frequency_penalty
    if input_data.presence_penalty is not None:
        body["presence_penalty"] = input_data.presence_penalty
    if input_data.seed is not None:
        body["seed"] = input_data.seed
    if input_data.stop:
        body["stop"] = input_data.stop

    # MiniMax does not document tool_choice or response_format on its
    # OpenAI-compatible endpoint. "auto" is the long-standing default and
    # "none" is honoured client-side by not sending tools; anything else is
    # rejected rather than sent on faith.
    choice = input_data.tool_choice
    if choice is not None and choice.mode in (ToolChoiceMode.REQUIRED, ToolChoiceMode.FUNCTION):
        raise ValueError(f"MiniMax does not support tool_choice mode '{choice.mode.value}'")
    if tools and not (choice is not None and choice.mode == ToolChoiceMode.NONE):
        body["tools"] = tools
        body["tool_choice"] = "auto"

    fmt = input_data.response_format
    if fmt is not None and fmt.type != ResponseFormatType.TEXT:
        raise ValueError(f"MiniMax does not support response_format type '{fmt.type.value}'")

    reasoning_effort = getattr(input_data, "reasoning_effort", None)
    if reasoning_effort and reasoning_effort != "none":
        reasoning_exclude = getattr(input_data, "reasoning_exclude", False)
        reasoning_config = {"exclude": reasoning_exclude, "effort": reasoning_effort}
        reasoning_max = getattr(input_data, "reasoning_max_tokens", None)
        if reasoning_max and reasoning_max > 0:
            reasoning_config["max_tokens"] = reasoning_max
        body["reasoning"] = reasoning_config

    return body


def _create_initial_state() -> Dict[str, Any]:
    return {
        "raw_content": "",
        "response": "",
        "reasoning": "",
        "reasoning_field": "",
        "reasoning_details": [],
        "tool_calls": [],
        "input_tokens": 0,
        "output_tokens": 0,
    }


_THINK_OPEN, _THINK_CLOSE = "<think>", "</think>"


def _split_think(raw: str) -> Tuple[str, str]:
    """Split MiniMax content into (response, thinking).

    M3 emits reasoning inline as <think>...</think> in `content` in addition
    to the reasoning_content field. Runs over the full accumulated content on
    every chunk, so tags split across chunk boundaries resolve naturally. A
    trailing partial tag (e.g. "<thi") is held back from response until the
    next chunk decides what it is, keeping response deltas append-only.
    """
    response, thinking = [], []
    i = 0
    while True:
        start = raw.find(_THINK_OPEN, i)
        if start == -1:
            tail = raw[i:]
            # Hold back a suffix that could be the beginning of a tag.
            for k in range(min(len(tail), len(_THINK_CLOSE) - 1), 0, -1):
                if _THINK_OPEN.startswith(tail[-k:]) or _THINK_CLOSE.startswith(tail[-k:]):
                    tail = tail[:-k]
                    break
            response.append(tail)
            break
        response.append(raw[i:start])
        end = raw.find(_THINK_CLOSE, start + len(_THINK_OPEN))
        if end == -1:
            thinking.append(raw[start + len(_THINK_OPEN):])  # still thinking
            break
        thinking.append(raw[start + len(_THINK_OPEN):end])
        i = end + len(_THINK_CLOSE)
    text, think = "".join(response), "".join(thinking)
    if think:
        text = text.lstrip("\n")  # newline MiniMax emits between </think> and the answer
    return text, think


def _increment(prev: str, cur: str) -> str:
    return cur[len(prev):] if cur.startswith(prev) else ""


def _parse_sse_chunk(data: Dict[str, Any], state: Dict[str, Any]) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Fold one SSE chunk into state.
    Returns (finish_reason, delta_dict); delta_dict is LLMDelta-shaped or None."""
    error = data.get("error")
    if error:
        msg = error.get("message", str(error)) if isinstance(error, dict) else str(error)
        raise RuntimeError(f"MiniMax mid-stream error: {msg}")

    usage = data.get("usage")
    if usage:
        if usage.get("prompt_tokens") is not None:
            state["input_tokens"] = usage["prompt_tokens"]
        if usage.get("completion_tokens") is not None:
            state["output_tokens"] = usage["completion_tokens"]
        completion_details = usage.get("completion_tokens_details")
        if completion_details:
            reasoning_tokens = completion_details.get("reasoning_tokens")
            if reasoning_tokens is not None:
                state["reasoning_tokens"] = reasoning_tokens

    choices = data.get("choices", [])
    if not choices:
        return None, None

    choice = choices[0]
    finish_reason = choice.get("finish_reason")
    delta = choice.get("delta", {})
    llm_delta: Dict[str, Any] = {}

    reasoning = delta.get("reasoning") or delta.get("reasoning_content")
    if reasoning:
        state["reasoning_field"] += reasoning

    content = delta.get("content")
    if content:
        state["raw_content"] += content

    if reasoning or content:
        response, thinking = _split_think(state["raw_content"])
        inc = _increment(state["response"], response)
        if inc:
            llm_delta["response"] = inc
        state["response"] = response

        # reasoning_content is authoritative when the provider sends it;
        # the inline block is only used when it is the sole source.
        new_reasoning = state["reasoning_field"] or thinking
        inc = _increment(state["reasoning"], new_reasoning)
        if inc:
            llm_delta["reasoning"] = inc
        state["reasoning"] = new_reasoning

    reasoning_details = delta.get("reasoning_details")
    if reasoning_details:
        state["reasoning_details"].extend(reasoning_details)

    tool_calls = delta.get("tool_calls")
    if tool_calls:
        for tc in tool_calls:
            _process_tool_call_delta(tc, state["tool_calls"])
        llm_delta["tool_calls"] = tool_calls

    return finish_reason, llm_delta if llm_delta else None


def _process_tool_call_delta(tc: Dict[str, Any], tool_calls: List[Dict[str, Any]]) -> None:
    tool_id = tc.get("id")
    if tool_id:
        current = next((t for t in tool_calls if t["id"] == tool_id), None)
        if not current:
            current = {"id": tool_id, "type": "function", "function": {"name": "", "arguments": ""}}
            tool_calls.append(current)
    else:
        current = tool_calls[-1] if tool_calls else None

    fn = tc.get("function", {})
    if current and fn:
        if fn.get("name"):
            current["function"]["name"] = fn["name"]
        if fn.get("arguments"):
            current["function"]["arguments"] += fn["arguments"]


def _build_output(state: Dict[str, Any]) -> Dict[str, Any]:
    out = {"response": state["response"]}
    if state["reasoning"]:
        out["reasoning"] = state["reasoning"]
    if state["reasoning_details"]:
        out["reasoning_details"] = state["reasoning_details"]
    if state["tool_calls"]:
        out["tool_calls"] = state["tool_calls"]

    inputs, outputs = [], []
    if state.get("input_tokens"):
        inputs.append(TextMeta(tokens=state["input_tokens"]))
    if state.get("output_tokens"):
        outputs.append(TextMeta(tokens=state["output_tokens"]))

    if inputs or outputs:
        out["output_meta"] = OutputMeta(inputs=inputs, outputs=outputs)

    return out

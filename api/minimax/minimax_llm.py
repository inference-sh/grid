"""MiniMax LLM streaming helper — raw httpx SSE against OpenAI-compatible endpoint."""

from __future__ import annotations

import asyncio
import json
import time
from typing import List, Optional, Dict, Any, AsyncGenerator

import httpx

from inferencesh import OutputMeta, TextMeta
from inferencesh.models.llm import build_openai_messages, build_tools

MINIMAX_BASE_URL = "https://api.minimax.io/v1"
STREAM_SILENCE_TIMEOUT = 60


async def stream_completion(api_key: str, input_data, model: str) -> AsyncGenerator[Dict[str, Any], None]:
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
                _parse_sse_chunk(data, state)
                yield _build_output(state)

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

    ms = getattr(input_data, "model_settings", None)

    def _get(name, default=None):
        if ms is not None:
            v = getattr(ms, name, None)
            if v is not None:
                return v
        return getattr(input_data, name, default)

    if _get("temperature") is not None:
        body["temperature"] = _get("temperature")
    if _get("top_p") is not None:
        body["top_p"] = _get("top_p")
    top_k = _get("top_k")
    if top_k is not None and top_k >= 0:
        body["top_k"] = top_k
    freq_pen = _get("frequency_penalty")
    if freq_pen is not None:
        body["frequency_penalty"] = freq_pen
    pres_pen = _get("presence_penalty")
    if pres_pen is not None:
        body["presence_penalty"] = pres_pen
    seed = _get("seed")
    if seed is not None:
        body["seed"] = seed
    user_stop = _get("stop")
    if user_stop:
        body["stop"] = user_stop
    ms_max = _get("max_tokens") if ms else None
    if ms_max is not None:
        body["max_tokens"] = ms_max

    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"

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
        "response": "",
        "reasoning": "",
        "reasoning_details": [],
        "tool_calls": [],
        "input_tokens": 0,
        "output_tokens": 0,
    }


def _parse_sse_chunk(data: Dict[str, Any], state: Dict[str, Any]) -> Optional[str]:
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
        return None

    choice = choices[0]
    finish_reason = choice.get("finish_reason")
    delta = choice.get("delta", {})

    content = delta.get("content")
    if content:
        state["response"] += content

    reasoning = delta.get("reasoning") or delta.get("reasoning_content")
    if reasoning:
        state["reasoning"] += reasoning

    reasoning_details = delta.get("reasoning_details")
    if reasoning_details:
        state["reasoning_details"].extend(reasoning_details)

    tool_calls = delta.get("tool_calls")
    if tool_calls:
        for tc in tool_calls:
            _process_tool_call_delta(tc, state["tool_calls"])

    return finish_reason


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

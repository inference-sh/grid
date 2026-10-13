"""Opper stream processing helpers for LLM inference."""

import asyncio
from typing import List, Optional, Dict, Any, AsyncGenerator, Tuple
from inferencesh import File
from inferencesh.models.llm import LLMUsage, build_openai_messages, build_tools, merge_reasoning_details, openai_response_format, openai_tool_choice
from inferencesh import OutputMeta
from inferencesh.models.output_meta import RawMeta

# This app's entry in an output's reasoning_state: the reasoning_details Opper
# returns (OpenRouter's shape), sent back as they came.
REASONING_STATE_KEY = "opper"


def _plain_details(details) -> List[Dict[str, Any]]:
    """reasoning_details as the OpenAI SDK hands them (an extra field: dicts,
    or models where it parsed them) as plain dicts."""
    return [d.model_dump(exclude_none=True) if hasattr(d, "model_dump") else d for d in details or []]


def get_reasoning_config(input_data) -> Optional[Dict[str, Any]]:
    """Build reasoning config for Opper API."""
    reasoning_effort = getattr(input_data, "reasoning_effort", None)
    reasoning_max_tokens = getattr(input_data, "reasoning_max_tokens", None)
    reasoning_exclude = getattr(input_data, "reasoning_exclude", False)

    if reasoning_effort == "none" and not reasoning_max_tokens:
        return None

    reasoning_config = {"exclude": reasoning_exclude}

    if reasoning_max_tokens is not None and reasoning_max_tokens > 0:
        reasoning_config["max_tokens"] = reasoning_max_tokens
    elif reasoning_effort and reasoning_effort != "none":
        reasoning_config["effort"] = reasoning_effort
    else:
        return None

    return reasoning_config


def handle_api_error(e: Exception, prefix: str = "Opper API") -> RuntimeError:
    """Extract error message from API exception."""
    if hasattr(e, "response") and e.response is not None:
        try:
            import json
            error_data = e.response.json()
            error_obj = error_data.get("error", {})
            msg = error_obj.get("message", str(e))

            headers = e.response.headers if hasattr(e.response, "headers") else {}
            request_id = headers.get("x-request-id", "")

            metadata = error_obj.get("metadata", {})
            raw = metadata.get("raw")
            if raw:
                try:
                    raw_error = json.loads(raw)
                    nested_msg = raw_error.get("error", {}).get("message")
                    if nested_msg:
                        provider_name = metadata.get("provider_name", "Provider")
                        return RuntimeError(f"{prefix} error ({provider_name}): {nested_msg} [req:{request_id}]")
                except json.JSONDecodeError:
                    pass

            import logging
            logging.warning(f"{prefix} error: {msg} | request_id={request_id} | body={json.dumps(error_data)}")

            return RuntimeError(f"{prefix} error: {msg} [req:{request_id}]")
        except Exception:
            pass
    return RuntimeError(f"{prefix} error: {str(e)}")


def check_chunk_error(chunk, prefix: str = "Opper") -> None:
    """Raise if chunk contains an error."""
    if hasattr(chunk, "error") and chunk.error:
        msg = chunk.error.get("message", "Unknown error") if isinstance(chunk.error, dict) else str(chunk.error)
        raise RuntimeError(f"{prefix} mid-stream error: {msg}")

    if chunk.choices and len(chunk.choices) > 0:
        if getattr(chunk.choices[0], "finish_reason", None) == "error":
            raise RuntimeError(f"{prefix} stream terminated due to error")


def process_tool_call_delta(delta, tool_calls: List[Dict[str, Any]]) -> None:
    """Process a tool call delta and update the tool_calls list in place."""
    tool_id = delta.id
    if tool_id:
        current = next((t for t in tool_calls if t["id"] == tool_id), None)
        if not current:
            current = {"id": tool_id, "type": "function", "function": {"name": "", "arguments": ""}}
            tool_calls.append(current)
    else:
        current = tool_calls[-1] if tool_calls else None

    if current and delta.function:
        if delta.function.name:
            current["function"]["name"] = delta.function.name
        if delta.function.arguments:
            current["function"]["arguments"] += delta.function.arguments


def _plain(obj) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    return obj.model_dump() if hasattr(obj, "model_dump") else dict(vars(obj))


def read_usage(usage, state: Dict[str, Any]) -> None:
    """Fold Opper's usage into state.

    Opper reports cache use the way Anthropic does (probed 2026-10-13): with
    cache_control, prompt_tokens counts only the uncached input, and
    prompt_tokens_details carries cached_tokens (reads) and cache_write_tokens
    (writes) beside it. LLMUsage.prompt_tokens is the whole input, so the
    three are added. usage.cost is Opper's charge in dollars (tokens and any
    tool fees such as web search)."""
    u = _plain(usage)
    if not u:
        return
    details = _plain(u.get("prompt_tokens_details"))
    cached = details.get("cached_tokens") or 0
    written = details.get("cache_write_tokens") or details.get("cache_creation_tokens") or 0
    if u.get("prompt_tokens") is not None:
        state["input_tokens"] = u["prompt_tokens"] + cached + written
        state["cached_tokens"] = cached
        state["cache_write_tokens"] = written
    if u.get("completion_tokens") is not None:
        state["output_tokens"] = u["completion_tokens"]
    reasoning = _plain(u.get("completion_tokens_details")).get("reasoning_tokens")
    if reasoning is not None:
        state["reasoning_tokens"] = reasoning
    if u.get("cost") is not None:
        state["cost_usd"] = u["cost"]


def process_chunk(chunk, state: Dict[str, Any]) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Fold one chunk into state.
    Returns (finish_reason, delta_dict); delta_dict is LLMDelta-shaped or None."""
    check_chunk_error(chunk)

    read_usage(getattr(chunk, "usage", None), state)

    if not chunk.choices:
        return None, None

    delta = chunk.choices[0].delta
    finish_reason = chunk.choices[0].finish_reason
    llm_delta: Dict[str, Any] = {}

    if delta.content:
        state["response"] += delta.content
        llm_delta["response"] = delta.content

    if hasattr(delta, "reasoning") and delta.reasoning:
        state["reasoning"] += delta.reasoning
        llm_delta["reasoning"] = delta.reasoning

    if getattr(delta, "reasoning_details", None):
        state["reasoning_details"] = merge_reasoning_details(state["reasoning_details"] + _plain_details(delta.reasoning_details))

    if delta.tool_calls:
        for tc in delta.tool_calls:
            process_tool_call_delta(tc, state["tool_calls"])
        llm_delta["tool_calls"] = [
            tc.model_dump(exclude_none=True) if hasattr(tc, "model_dump") else tc for tc in delta.tool_calls
        ]

    return finish_reason, llm_delta if llm_delta else None


def build_output(state: Dict[str, Any]) -> Dict[str, Any]:
    """Build output dict from accumulated state."""
    out = {"response": state["response"]}
    if state["reasoning"]:
        out["reasoning"] = state["reasoning"]
    if state["reasoning_details"]:
        out["reasoning_state"] = {REASONING_STATE_KEY: state["reasoning_details"]}
    if state["tool_calls"]:
        out["tool_calls"] = state["tool_calls"]

    # Token usage, cache reads and writes included; the token report is built
    # from it. Opper's own charge follows as a raw input.
    meta = OutputMeta()
    if state["input_tokens"] or state["output_tokens"]:
        out["usage"] = LLMUsage(
            prompt_tokens=state["input_tokens"],
            cached_tokens=state["cached_tokens"],
            cache_write_tokens=state["cache_write_tokens"],
            completion_tokens=state["output_tokens"],
            total_tokens=state["input_tokens"] + state["output_tokens"],
            reasoning_tokens=state["reasoning_tokens"],
        )
        meta = OutputMeta.from_llm_usage(out["usage"])
    if state.get("cost_usd") is not None:
        meta.inputs.append(RawMeta(cost=state["cost_usd"] * 100))
    if meta.inputs or meta.outputs:
        out["output_meta"] = meta

    return out


def create_initial_state() -> Dict[str, Any]:
    """Create initial state dict for stream processing."""
    return {
        "response": "",
        "reasoning": "",
        "reasoning_details": [],
        "tool_calls": [],
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "reasoning_tokens": 0,
        "cost_usd": None,
    }


def _convert_files_to_documents(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert OpenAI file content blocks to Anthropic document format.

    Opper passes through to Anthropic natively, so file blocks need to be
    in Anthropic's document format rather than OpenAI's file format.
    """
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for i, part in enumerate(content):
            if part.get("type") == "file":
                file_data = part.get("file", {})
                file_uri = file_data.get("file_data", "")
                if file_uri.startswith("http://") or file_uri.startswith("https://"):
                    content[i] = {"type": "document", "source": {"type": "url", "url": file_uri}}
                elif file_uri.startswith("data:"):
                    import re
                    m = re.match(r"data:([^;]+);base64,(.*)", file_uri, re.DOTALL)
                    if m:
                        content[i] = {"type": "document", "source": {"type": "base64", "media_type": m.group(1), "data": m.group(2)}}
                else:
                    content[i] = {"type": "document", "source": {"type": "url", "url": file_uri}}
    return messages


def _build_params(input_data, model: str, stream: bool) -> Dict[str, Any]:
    """Build common request parameters."""
    messages = build_openai_messages(input_data, file_mode="url", image_mode="url", reasoning_state_key=REASONING_STATE_KEY)
    messages = _convert_files_to_documents(messages)
    tools = build_tools(input_data.tools) if input_data.tools else None

    params = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "max_tokens": 64000,
    }

    if stream:
        params["stream_options"] = {"include_usage": True}

    # Opper's compat endpoint declares tool_choice and response_format in its
    # OpenAPI schema; pass them through in OpenAI spelling and let the
    # upstream provider reject what it cannot honour.
    if tools:
        params["tools"] = tools
        params["tool_choice"] = openai_tool_choice(input_data.tool_choice)

    response_format = openai_response_format(input_data.response_format)
    if response_format is not None:
        params["response_format"] = response_format

    extra_body = {}
    reasoning_config = get_reasoning_config(input_data)
    if reasoning_config:
        extra_body["reasoning"] = reasoning_config

    if extra_body:
        params["extra_body"] = extra_body

    return params


async def complete(client, input_data, model: str) -> Dict[str, Any]:
    """Non-streaming completion from Opper."""
    params = _build_params(input_data, model, stream=False)

    try:
        response = await asyncio.wait_for(client.chat.completions.create(**params), timeout=120.0)
    except asyncio.TimeoutError:
        raise RuntimeError("Opper API call timed out after 120 seconds")
    except Exception as e:
        raise handle_api_error(e)

    state = create_initial_state()
    message = response.choices[0].message

    if message.content:
        state["response"] = message.content

    if hasattr(message, "reasoning") and message.reasoning:
        state["reasoning"] = message.reasoning

    if getattr(message, "reasoning_details", None):
        state["reasoning_details"] = merge_reasoning_details(_plain_details(message.reasoning_details))

    if message.tool_calls:
        for tc in message.tool_calls:
            state["tool_calls"].append({
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments}
            })

    read_usage(getattr(response, "usage", None), state)

    return build_output(state)


async def stream_completion(
    client, input_data, model: str, *, with_deltas: bool = False,
) -> AsyncGenerator[Any, None]:
    """Stream a completion from Opper.

    with_deltas=False (default): yields accumulated output dicts.
    with_deltas=True: yields (output_dict, delta_dict | None) tuples, where
    delta_dict has LLMDelta-shaped keys (response, reasoning, tool_calls).
    """
    params = _build_params(input_data, model, stream=True)

    try:
        stream = await asyncio.wait_for(client.chat.completions.create(**params), timeout=15.0)
    except asyncio.TimeoutError:
        raise RuntimeError("Opper API call timed out after 15 seconds")
    except Exception as e:
        raise handle_api_error(e)

    state = create_initial_state()
    last_chunk_time = asyncio.get_event_loop().time()

    try:
        async for chunk in stream:
            now = asyncio.get_event_loop().time()
            if now - last_chunk_time > 120.0:
                raise RuntimeError("Stream timed out - no chunks received for 120 seconds")
            last_chunk_time = now

            _, chunk_delta = process_chunk(chunk, state)
            output = build_output(state)
            yield (output, chunk_delta) if with_deltas else output
    finally:
        if hasattr(stream, "aclose"):
            await stream.aclose()

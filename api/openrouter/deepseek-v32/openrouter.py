"""OpenRouter stream processing helpers for LLM inference."""

import asyncio
from typing import List, Optional, Dict, Any, AsyncGenerator
from inferencesh import File
from inferencesh.models.llm import build_messages, build_tools


def get_reasoning_config(input_data) -> Optional[Dict[str, Any]]:
    """Build reasoning config for OpenRouter API."""
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


def handle_api_error(e: Exception, prefix: str = "OpenRouter API") -> RuntimeError:
    """Extract error message from API exception."""
    if hasattr(e, "response") and e.response is not None:
        try:
            error_data = e.response.json()
            msg = error_data.get("error", {}).get("message", str(e))
            return RuntimeError(f"{prefix} error: {msg}")
        except Exception:
            pass
    return RuntimeError(f"{prefix} error: {str(e)}")


def check_chunk_error(chunk, prefix: str = "OpenRouter") -> None:
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


def process_chunk(chunk, state: Dict[str, Any]) -> Optional[str]:
    """Process a single chunk and update state dict. Returns finish_reason if present."""
    check_chunk_error(chunk)
    
    delta = chunk.choices[0].delta
    finish_reason = chunk.choices[0].finish_reason

    if delta.content:
        state["response"] += delta.content

    if hasattr(delta, "reasoning") and delta.reasoning:
        state["reasoning"] += delta.reasoning

    if hasattr(delta, "reasoning_details") and delta.reasoning_details:
        state["reasoning_details"].extend(delta.reasoning_details)

    if delta.tool_calls:
        for tc in delta.tool_calls:
            process_tool_call_delta(tc, state["tool_calls"])

    if hasattr(delta, "images") and delta.images:
        for img in delta.images:
            url = img.get("image_url", {}).get("url") if isinstance(img, dict) else None
            if url and url not in state["image_urls"]:
                state["image_urls"].append(url)

    return finish_reason


def build_output(state: Dict[str, Any]) -> Dict[str, Any]:
    """Build output dict from accumulated state."""
    out = {"response": state["response"]}
    if state["reasoning"]:
        out["reasoning"] = state["reasoning"]
    if state["reasoning_details"]:
        out["reasoning_details"] = state["reasoning_details"]
    if state["tool_calls"]:
        out["tool_calls"] = state["tool_calls"]
    if state["image_urls"]:
        out["images"] = [File(uri=url) for url in state["image_urls"]]
    return out


def create_initial_state() -> Dict[str, Any]:
    """Create initial state dict for stream processing."""
    return {
        "response": "",
        "reasoning": "",
        "reasoning_details": [],
        "tool_calls": [],
        "image_urls": [],
    }


async def stream_completion(client, input_data, model: str) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Stream a completion from OpenRouter and yield output dicts.
    
    Handles: message building, tools, reasoning config, timeouts, error handling.
    """
    messages = build_messages(input_data)
    tools = build_tools(input_data.tools) if input_data.tools else None

    params = {
        "model": model,
        "messages": messages,
        "stream": True,
        "extra_headers": {"HTTP-Referer": "https://inference.sh", "X-Title": "inference.sh"},
        "stop": ["<end_of_turn>", "<eos>", "<|im_end|>"],
        "max_tokens": 64000,
    }

    if tools:
        params["tools"] = tools
        params["tool_choice"] = "auto"

    reasoning_config = get_reasoning_config(input_data)
    if reasoning_config:
        params["extra_body"] = {"reasoning": reasoning_config}

    try:
        stream = await asyncio.wait_for(client.chat.completions.create(**params), timeout=15.0)
    except asyncio.TimeoutError:
        raise RuntimeError("OpenRouter API call timed out after 15 seconds")
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

            finish_reason = process_chunk(chunk, state)
            yield build_output(state)

            if finish_reason:
                break
    finally:
        if hasattr(stream, "aclose"):
            await stream.aclose()


"""Anthropic API helper for Claude model inference.

Shared helper for all Claude model apps. Handles message conversion,
streaming, tool use, and token usage tracking via the native Anthropic SDK.
"""

import asyncio
import json
import base64
import re
import logging
from typing import AsyncGenerator, List, Optional, Dict, Any, Tuple

from inferencesh import OutputMeta, TextMeta
from inferencesh.models.llm import (
    LLMInput,
    ContextMessageRole,
    ReasoningEffortEnum,
)

logger = logging.getLogger(__name__)

# Budget tokens mapping for reasoning effort levels
EFFORT_TO_BUDGET = {
    ReasoningEffortEnum.LOW: 1024,
    ReasoningEffortEnum.MEDIUM: 10240,
    ReasoningEffortEnum.HIGH: 32000,
}


def extract_base64_from_data_uri(data_uri: str) -> Tuple[str, str]:
    """Extract media type and base64 data from a data URI."""
    match = re.match(r"data:([^;]+);base64,(.+)", data_uri)
    if match:
        return match.group(1), match.group(2)
    return "image/png", data_uri


def _build_image_block(uri: str) -> Dict[str, Any]:
    """Build an Anthropic image content block from a URI (data: or https:)."""
    if uri.startswith("data:"):
        media_type, data = extract_base64_from_data_uri(uri)
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        }
    return {"type": "image", "source": {"type": "url", "url": uri}}


def _build_image_block_from_path(path: str) -> Dict[str, Any]:
    """Build an Anthropic image content block from a file path."""
    with open(path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")
    ext = path.lower().rsplit(".", 1)[-1]
    media_type = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "webp": "image/webp",
    }.get(ext, "image/png")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": image_data},
    }


def _collect_images(images) -> List[Dict[str, Any]]:
    """Convert a list of image objects to Anthropic content blocks."""
    blocks = []
    if not images:
        return blocks
    for image in images:
        if image.path:
            blocks.append(_build_image_block_from_path(image.path))
        elif image.uri:
            blocks.append(_build_image_block(image.uri))
    return blocks


def convert_messages_to_anthropic(
    input_data: LLMInput,
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """Convert LLMInput to Anthropic message format.

    Returns (system_prompt, messages) tuple.
    """
    system_prompt = input_data.system_prompt if input_data.system_prompt else None
    messages: List[Dict[str, Any]] = []

    for msg in input_data.context:
        role = "user" if msg.role == ContextMessageRole.USER else "assistant"

        # Tool result messages
        if msg.role == ContextMessageRole.TOOL and msg.tool_call_id:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id,
                            "content": msg.text or "",
                        }
                    ],
                }
            )
            continue

        content: List[Dict[str, Any]] = []

        if msg.text:
            content.append({"type": "text", "text": msg.text})

        content.extend(_collect_images(getattr(msg, "images", None)))

        # Assistant tool_use blocks
        if msg.role == ContextMessageRole.ASSISTANT and msg.tool_calls:
            for tc in msg.tool_calls:
                func = tc.get("function", {})
                args_str = func.get("arguments", "{}")
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                except json.JSONDecodeError:
                    args = {}
                content.append(
                    {
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": func.get("name", ""),
                        "input": args,
                    }
                )

        if content:
            messages.append({"role": role, "content": content})

    # Current user input
    user_content: List[Dict[str, Any]] = []
    if hasattr(input_data, "text") and input_data.text:
        user_content.append({"type": "text", "text": input_data.text})
    user_content.extend(_collect_images(getattr(input_data, "images", None)))

    # Tool result input
    if hasattr(input_data, "role") and input_data.role == ContextMessageRole.TOOL:
        if hasattr(input_data, "tool_call_id") and input_data.tool_call_id:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": input_data.tool_call_id,
                            "content": input_data.text or "",
                        }
                    ],
                }
            )
        elif user_content:
            messages.append({"role": "user", "content": user_content})
    elif user_content:
        messages.append({"role": "user", "content": user_content})

    # Merge consecutive same-role messages (Anthropic requirement)
    merged: List[Dict[str, Any]] = []
    for msg in messages:
        if merged and merged[-1]["role"] == msg["role"]:
            prev = merged[-1]["content"]
            curr = msg["content"]
            if isinstance(prev, list) and isinstance(curr, list):
                prev.extend(curr)
            elif isinstance(prev, str) and isinstance(curr, str):
                merged[-1]["content"] = prev + "\n" + curr
            else:
                prev_list = prev if isinstance(prev, list) else [{"type": "text", "text": prev}]
                curr_list = curr if isinstance(curr, list) else [{"type": "text", "text": curr}]
                merged[-1]["content"] = prev_list + curr_list
        else:
            merged.append(msg)

    return system_prompt, merged


def convert_tools_to_anthropic(
    tools: Optional[List[Dict[str, Any]]],
) -> Optional[List[Dict[str, Any]]]:
    """Convert OpenAI-format tools to Anthropic format."""
    if not tools:
        return None
    result = []
    for tool in tools:
        func_def = tool.get("function", tool) if "type" in tool else tool
        result.append(
            {
                "name": func_def.get("name", ""),
                "description": func_def.get("description", ""),
                "input_schema": func_def.get("parameters", {"type": "object", "properties": {}}),
            }
        )
    return result


def build_thinking_param(input_data) -> Dict[str, Any]:
    """Build the thinking parameter for Anthropic API based on reasoning config."""
    reasoning_max = getattr(input_data, "reasoning_max_tokens", None)
    reasoning_effort = getattr(input_data, "reasoning_effort", None)

    if reasoning_max is None or reasoning_max == 0:
        return {"type": "disabled"}

    if reasoning_effort and reasoning_effort != ReasoningEffortEnum.NONE:
        if reasoning_max is not None and reasoning_max > 1024:
            budget = reasoning_max
        else:
            budget = EFFORT_TO_BUDGET.get(reasoning_effort, 1024)
        return {"type": "enabled", "budget_tokens": budget}

    return {"type": "disabled"}


def handle_api_error(e: Exception, prefix: str = "Anthropic API") -> RuntimeError:
    """Extract error message from API exception."""
    if hasattr(e, "response") and e.response is not None:
        try:
            error_data = e.response.json()
            msg = error_data.get("error", {}).get("message", str(e))
            return RuntimeError(f"{prefix} error: {msg}")
        except Exception:
            pass
    return RuntimeError(f"{prefix} error: {str(e)}")


def create_initial_state() -> Dict[str, Any]:
    """Create initial state dict for stream processing."""
    return {
        "response": "",
        "thinking": "",
        "tool_calls": [],
        "current_tool": None,
        "input_tokens": 0,
        "output_tokens": 0,
    }


def build_output(state: Dict[str, Any]) -> Dict[str, Any]:
    """Build output dict from accumulated state."""
    out: Dict[str, Any] = {"response": state["response"]}
    if state["thinking"]:
        out["reasoning"] = state["thinking"]
    if state["tool_calls"]:
        out["tool_calls"] = state["tool_calls"]

    inputs = []
    outputs = []
    if state.get("input_tokens"):
        inputs.append(TextMeta(tokens=state["input_tokens"]))
    if state.get("output_tokens"):
        outputs.append(TextMeta(tokens=state["output_tokens"]))
    if inputs or outputs:
        out["output_meta"] = OutputMeta(inputs=inputs, outputs=outputs)

    return out


def build_params(
    input_data,
    model: str,
    max_tokens: int = 64000,
) -> Dict[str, Any]:
    """Build common request parameters for Anthropic API."""
    system_prompt, messages = convert_messages_to_anthropic(input_data)
    tools = convert_tools_to_anthropic(input_data.tools) if input_data.tools else None

    params: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
    }

    if system_prompt:
        params["system"] = system_prompt

    if tools:
        params["tools"] = tools

    params["thinking"] = build_thinking_param(input_data)

    return params


async def stream_completion(
    client, input_data, model: str, max_tokens: int = 64000
) -> AsyncGenerator[Dict[str, Any], None]:
    """Stream a completion from Anthropic API and yield output dicts."""
    params = build_params(input_data, model, max_tokens)

    logger.info(f"Calling Anthropic API: model={model}, messages={len(params['messages'])}")

    try:
        stream = await asyncio.wait_for(
            client.messages.create(**params), timeout=30.0
        )
    except asyncio.TimeoutError:
        raise RuntimeError("Anthropic API call timed out after 30 seconds")
    except Exception as e:
        raise handle_api_error(e)

    state = create_initial_state()
    last_chunk_time = asyncio.get_event_loop().time()

    try:
        async for event in stream:
            now = asyncio.get_event_loop().time()
            if now - last_chunk_time > 120.0:
                raise RuntimeError("Stream timed out - no events for 120 seconds")
            last_chunk_time = now

            if event.type == "content_block_start":
                block = event.content_block
                if block.type == "tool_use":
                    state["current_tool"] = {
                        "id": block.id,
                        "type": "function",
                        "function": {"name": block.name, "arguments": ""},
                    }
                    state["tool_calls"].append(state["current_tool"])

            elif event.type == "content_block_delta":
                delta = event.delta
                if delta.type == "text_delta":
                    state["response"] += delta.text
                elif delta.type == "thinking_delta":
                    state["thinking"] += delta.thinking
                elif delta.type == "input_json_delta":
                    if state["current_tool"]:
                        state["current_tool"]["function"]["arguments"] += delta.partial_json

            elif event.type == "content_block_stop":
                state["current_tool"] = None

            elif event.type == "message_delta":
                usage = getattr(event, "usage", None)
                if usage:
                    state["output_tokens"] = getattr(usage, "output_tokens", 0)

            elif event.type == "message_start":
                msg = getattr(event, "message", None)
                if msg:
                    usage = getattr(msg, "usage", None)
                    if usage:
                        state["input_tokens"] = getattr(usage, "input_tokens", 0)

            yield build_output(state)

    except Exception as e:
        if "overloaded" in str(e).lower():
            raise RuntimeError("Anthropic API is overloaded, please try again later")
        raise

    logger.info(
        f"Anthropic stream complete: in={state['input_tokens']} out={state['output_tokens']}"
    )


async def complete(
    client, input_data, model: str, max_tokens: int = 64000
) -> Dict[str, Any]:
    """Non-streaming completion from Anthropic API."""
    params = build_params(input_data, model, max_tokens)
    params["stream"] = False

    logger.info(f"Calling Anthropic API (non-stream): model={model}")

    try:
        response = await asyncio.wait_for(
            client.messages.create(**params), timeout=120.0
        )
    except asyncio.TimeoutError:
        raise RuntimeError("Anthropic API call timed out after 120 seconds")
    except Exception as e:
        raise handle_api_error(e)

    state = create_initial_state()

    # Extract usage
    if hasattr(response, "usage") and response.usage:
        state["input_tokens"] = getattr(response.usage, "input_tokens", 0)
        state["output_tokens"] = getattr(response.usage, "output_tokens", 0)

    # Process content blocks
    for block in response.content:
        if block.type == "text":
            state["response"] += block.text
        elif block.type == "thinking":
            state["thinking"] += block.thinking
        elif block.type == "tool_use":
            state["tool_calls"].append(
                {
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input),
                    },
                }
            )

    logger.info(
        f"Anthropic complete: in={state['input_tokens']} out={state['output_tokens']}"
    )

    return build_output(state)

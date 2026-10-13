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

from inferencesh import OutputMeta
from inferencesh.models.llm import LLMUsage
from inferencesh.images import ImageLimits, fit_image
from inferencesh.llm_types_gen import (
    ResponseFormat,
    ResponseFormatType,
    ToolChoice,
    ToolChoiceMode,
)
from inferencesh.models.llm import (
    LLMInput,
    ContextMessageRole,
    ReasoningEffortEnum,
)

logger = logging.getLogger(__name__)

# Non-streaming request limits. Streaming has no such cap.
NON_STREAM_TIMEOUT = 120.0
NON_STREAM_MAX_TOKENS = 16000

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


# Claude's vision limits (platform.claude.com/docs/en/build-with-claude/vision).
# A larger image is downscaled by the API anyway, so it is fitted here first:
# the request carries no bytes the model never sees. Claude 4.7 and later read
# up to a 2576px long edge and 4784 visual tokens (28x28px patches), earlier
# models 1568px and 1568 tokens. Each image is at most 10MB base64 (7.5MB of
# bytes), and a request with more than 20 images takes none past 2000px.
_HIGH_RES_MODELS = ("claude-opus-4-7", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5", "claude-fable-5", "claude-mythos-5")
_MAX_IMAGE_BYTES = 7_500_000
_MANY_IMAGES = 20
_MANY_IMAGES_LONG_EDGE = 2000


def image_limits(model: str, image_count: int) -> ImageLimits:
    """What one image in a request to model may be."""
    if model.startswith(_HIGH_RES_MODELS):
        long_edge, pixels = 2576, 4784 * 28 * 28
    else:
        long_edge, pixels = 1568, 1568 * 28 * 28
    if image_count > _MANY_IMAGES:
        long_edge = min(long_edge, _MANY_IMAGES_LONG_EDGE)
    return ImageLimits(max_long_edge=long_edge, max_pixels=pixels, max_bytes=_MAX_IMAGE_BYTES)


def _count_images(input_data) -> int:
    count = len(getattr(input_data, "images", None) or [])
    for msg in input_data.context or []:
        count += len(getattr(msg, "images", None) or [])
    return count


def _build_image_block_from_path(path: str, limits: Optional[ImageLimits] = None) -> Dict[str, Any]:
    """Build an Anthropic image content block from a file path, fitted to limits."""
    data, media_type = fit_image(path, limits or ImageLimits())
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": base64.b64encode(data).decode("utf-8")},
    }


def _collect_images(images, limits: Optional[ImageLimits] = None) -> List[Dict[str, Any]]:
    """Convert a list of image objects to Anthropic content blocks."""
    blocks = []
    if not images:
        return blocks
    for image in images:
        if image.path:
            blocks.append(_build_image_block_from_path(image.path, limits))
        elif image.uri:
            blocks.append(_build_image_block(image.uri))
    return blocks


def _build_tool_result_block(tool_use_id: str, text: Optional[str], images, limits: Optional[ImageLimits] = None) -> Dict[str, Any]:
    """Build a tool_result block. A result with images carries a content list
    (text block, then image blocks) so the model sees the images; a text-only
    result keeps the plain string content."""
    image_blocks = _collect_images(images, limits)
    if not image_blocks:
        return {"type": "tool_result", "tool_use_id": tool_use_id, "content": text or ""}
    content: List[Dict[str, Any]] = []
    if text:
        content.append({"type": "text", "text": text})
    content.extend(image_blocks)
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}


# The key this helper's entry has in a message's reasoning_state (INF-1066).
# Other apps' entries are carried by the platform and never read here.
REASONING_STATE_KEY = "anthropic"


def _replayed_thinking_blocks(msg) -> List[Dict[str, Any]]:
    """The thinking and redacted_thinking blocks an assistant message's
    response carried, from its reasoning_state entry; [] when it has none."""
    entry = (getattr(msg, "reasoning_state", None) or {}).get(REASONING_STATE_KEY)
    if not isinstance(entry, dict):
        return []
    return [dict(b) for b in entry.get("blocks") or [] if isinstance(b, dict)]


_THINKING_TYPES = ("thinking", "redacted_thinking")


def _reasoning_state(blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {REASONING_STATE_KEY: {"blocks": blocks}}


def convert_messages_to_anthropic(
    input_data: LLMInput,
    model: str = "",
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """Convert LLMInput to Anthropic message format, images fitted to model.

    Returns (system_prompt, messages) tuple.
    """
    limits = image_limits(model, _count_images(input_data))
    system_prompt = input_data.system_prompt if input_data.system_prompt else None
    messages: List[Dict[str, Any]] = []

    # Thinking blocks are replayed only for the assistant messages after the
    # last user turn: the tool loop still running. Anthropic strips an earlier
    # turn's thinking from the prompt anyway, so replaying it adds nothing,
    # and it is the only part of the history the prefix check can refuse: a
    # signature covers everything sent before its block, so compaction or a
    # changed tool list or system prompt before an old block is a 400 where
    # the check is enforced. Only a tool result continues the loop; any other
    # input is a new turn, and nothing is replayed.
    replay_from = len(input_data.context)
    if getattr(input_data, "role", None) == ContextMessageRole.TOOL:
        replay_from = next(
            (i + 1 for i in range(len(input_data.context) - 1, -1, -1)
             if input_data.context[i].role == ContextMessageRole.USER),
            0,
        )

    for i, msg in enumerate(input_data.context):
        role = "user" if msg.role == ContextMessageRole.USER else "assistant"

        # Tool result messages
        if msg.role == ContextMessageRole.TOOL and msg.tool_call_id:
            messages.append(
                {
                    "role": "user",
                    "content": [_build_tool_result_block(msg.tool_call_id, msg.text, getattr(msg, "images", None), limits)],
                }
            )
            continue

        content: List[Dict[str, Any]] = []

        # The model's thinking blocks come back first and exactly as it wrote
        # them: the signature covers them, and an edited or reordered block is
        # refused.
        if msg.role == ContextMessageRole.ASSISTANT and i >= replay_from:
            content.extend(_replayed_thinking_blocks(msg))

        # A whitespace-only text is the platform's stand-in for a turn with no
        # text (it keeps other providers from refusing an empty message); next
        # to the model's own blocks it is an edit of what the model wrote.
        if msg.text and not (content and not msg.text.strip()):
            content.append({"type": "text", "text": msg.text})

        content.extend(_collect_images(getattr(msg, "images", None), limits))

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
    user_content.extend(_collect_images(getattr(input_data, "images", None), limits))

    # Tool result input
    if hasattr(input_data, "role") and input_data.role == ContextMessageRole.TOOL:
        if hasattr(input_data, "tool_call_id") and input_data.tool_call_id:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        _build_tool_result_block(
                            input_data.tool_call_id, input_data.text, getattr(input_data, "images", None), limits
                        )
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
            if msg["role"] == "assistant" and isinstance(curr, list):
                # A message's thinking blocks lead its content, and merged
                # into the one before they would sit mid-message, where no
                # response put them. They are dropped; the first message's
                # stay where they were.
                curr = [b for b in curr if b.get("type") not in _THINKING_TYPES]
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


_COMBINATORS = ("anyOf", "oneOf", "allOf")


def normalize_input_schema(schema: Any) -> Dict[str, Any]:
    """Coerce a caller-supplied JSON Schema into one Anthropic accepts as input_schema.

    Anthropic requires top-level "type": "object" and rejects anyOf/oneOf/allOf at
    the top level. Callers send schemas from zod/pydantic/MCP servers that violate
    both, so: missing/invalid schemas become an empty object, top-level combinators
    are flattened into one object (properties merged; required is the intersection
    for anyOf/oneOf, the union for allOf), and a missing type is set to object.
    """
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}
    schema = dict(schema)

    for key in _COMBINATORS:
        branches = schema.pop(key, None)
        if not isinstance(branches, list):
            continue
        objects = [b for b in branches if isinstance(b, dict) and b.get("type") != "null"]
        properties = dict(schema.get("properties") or {})
        required_sets = []
        for branch in objects:
            properties.update(branch.get("properties") or {})
            required_sets.append(set(branch.get("required") or []))
        if properties:
            schema["properties"] = properties
        if required_sets:
            if key == "allOf":
                merged = set.union(*required_sets)
            else:
                merged = set.intersection(*required_sets)
            merged |= set(schema.get("required") or [])
            if merged:
                schema["required"] = sorted(merged)

    if schema.get("type") != "object":
        schema["type"] = "object"
    schema.setdefault("properties", {})
    return schema


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
                "input_schema": normalize_input_schema(func_def.get("parameters")),
            }
        )
    return result


# Models with always-on thinking: {"type": "disabled"} returns 400, and on the
# 5.1 / 5.5 models so does {"type": "enabled", "budget_tokens": N}. They take
# {"type": "adaptive"} (equivalent to omitting the parameter).
ALWAYS_ON_THINKING_MODELS = {
    "claude-fable-5", "claude-mythos-5",
    "claude-fable-5-1", "claude-mythos-5-1",
    "claude-opus-5-5",
}

# Models that removed thinking.budget_tokens (400 if sent). Thinking depth is
# controlled with adaptive thinking + output_config.effort instead.
EFFORT_MODELS = {
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-sonnet-5",
    # 5.1 / 5.5 generation: effort is the only thinking control.
    "claude-fable-5-1",
    "claude-mythos-5-1",
    "claude-opus-5-5",
    "claude-sonnet-5-5",
    "claude-haiku-5-5",
}

# Models where thinking is never turned off. On claude-opus-5 disabling it is
# a documented footgun: the model can emit a tool call as plain response text
# (the call silently never runs) and can leak <thinking> tags into the output.
# On the always-on models there is nothing to turn off. Either way a request
# with no reasoning asked for runs at the cheapest depth, output_config.effort
# "low", instead of {"type": "disabled"}.
NEVER_DISABLE_THINKING_MODELS = {
    "claude-opus-5",
    "claude-fable-5-1", "claude-mythos-5-1",
    "claude-opus-5-5",
}

# claude-sonnet-5-5 rejects {"type": "disabled"}; its lowest setting is
# {"type": "between_tools"} (no other field inside thinking, effort "high" or
# below, which every platform effort level is). The progress notes it writes
# between tool calls come back as thinking blocks.
BETWEEN_TOOLS_MODELS = {"claude-sonnet-5-5"}

# Models that return 400 on tool_choice "any" and "tool" (forced tool use),
# on count_tokens and Batches too. "auto" and "none" are unchanged. Refused
# here before the request: the platform has no prompt-side steering to offer
# in its place, and the schema-valid-arguments guarantee comes from the tool
# definition, not from forcing.
NO_FORCED_TOOL_CHOICE_MODELS = {
    "claude-fable-5-1", "claude-mythos-5-1",
    "claude-opus-5-5",
    "claude-sonnet-5-5",
}

# Reasoning effort -> output_config.effort level
EFFORT_TO_LEVEL = {
    ReasoningEffortEnum.LOW: "low",
    ReasoningEffortEnum.MEDIUM: "medium",
    ReasoningEffortEnum.HIGH: "high",
}


def _reasoning_requested(input_data) -> bool:
    """Whether the caller asked for reasoning (explicit budget + non-none effort)."""
    reasoning_max = getattr(input_data, "reasoning_max_tokens", None)
    reasoning_effort = getattr(input_data, "reasoning_effort", None)

    if reasoning_max is None or reasoning_max == 0:
        return False

    return bool(reasoning_effort) and reasoning_effort != ReasoningEffortEnum.NONE


# Adaptive thinking with a readable summary. Opus 4.7 and later default to
# display "omitted" (empty thinking text), so the summary is asked for
# explicitly; thinking is billed the same either way. On the always-on models
# {"type": "adaptive"} is equivalent to omitting the parameter and carries the
# display setting; between_tools takes no other field.
ADAPTIVE_THINKING = {"type": "adaptive", "display": "summarized"}


def build_thinking_param(input_data, model: str = "") -> Optional[Dict[str, Any]]:
    """Build the thinking parameter for Anthropic API based on reasoning config."""
    if model in ALWAYS_ON_THINKING_MODELS or model in NEVER_DISABLE_THINKING_MODELS:
        return ADAPTIVE_THINKING

    if not _reasoning_requested(input_data):
        if model in BETWEEN_TOOLS_MODELS:
            return {"type": "between_tools"}
        return {"type": "disabled"}

    if model in EFFORT_MODELS:
        return ADAPTIVE_THINKING

    reasoning_max = getattr(input_data, "reasoning_max_tokens", None)
    reasoning_effort = getattr(input_data, "reasoning_effort", None)
    if reasoning_max is not None and reasoning_max > 1024:
        budget = reasoning_max
    else:
        budget = EFFORT_TO_BUDGET.get(reasoning_effort, 1024)
    return {"type": "enabled", "budget_tokens": budget}


def build_output_config(input_data, model: str = "") -> Optional[Dict[str, Any]]:
    """Build the output_config parameter (effort) for models that support it."""
    if model not in EFFORT_MODELS:
        return None

    level = EFFORT_TO_LEVEL.get(getattr(input_data, "reasoning_effort", None))
    if level:
        return {"effort": level}

    if model in NEVER_DISABLE_THINKING_MODELS:
        # No reasoning requested: keep thinking on (disabling it misbehaves on
        # these models) but run at the cheapest depth.
        return {"effort": "low"}

    return None


def handle_api_error(e: Exception, prefix: str = "Anthropic API") -> RuntimeError:
    """Extract error message from API exception.

    Includes the status code and error type: without them a model the key has no
    access to reports as a bare `model: <id>`, which reads like the model does not
    exist rather than a 404 not_found_error on an existing but ungranted model.
    """
    status = getattr(e, "status_code", None)
    if hasattr(e, "response") and e.response is not None:
        status = status or getattr(e.response, "status_code", None)
        try:
            error = e.response.json().get("error", {})
            msg = error.get("message", str(e))
            detail = " ".join(str(p) for p in (status, error.get("type")) if p)
            return RuntimeError(f"{prefix} error ({detail}): {msg}" if detail else f"{prefix} error: {msg}")
        except Exception:
            pass
    return RuntimeError(f"{prefix} error ({status}): {e}" if status else f"{prefix} error: {e}")


def create_initial_state() -> Dict[str, Any]:
    """Create initial state dict for stream processing."""
    return {
        "response": "",
        "thinking": "",
        # thinking / redacted_thinking blocks as the API returned them, in
        # order, signatures included: sent back with the next request.
        "thinking_blocks": [],
        "current_thinking": None,
        "tool_calls": [],
        "current_tool": None,
        "input_tokens": 0,
        "output_tokens": 0,
        "stop_reason": "",
    }


# Anthropic stop_reason -> the platform's (OpenAI-style) stop reason.
_STOP_REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
}


def platform_stop_reason(stop_reason: Optional[str]) -> str:
    """Anthropic's stop_reason as LLMUsage.stop_reason; refusal and pause_turn
    pass through unchanged."""
    if not stop_reason:
        return ""
    return _STOP_REASONS.get(stop_reason, stop_reason)


def build_output(state: Dict[str, Any]) -> Dict[str, Any]:
    """Build output dict from accumulated state."""
    out: Dict[str, Any] = {"response": state["response"]}
    if state["thinking"]:
        out["reasoning"] = state["thinking"]
    if state["tool_calls"]:
        out["tool_calls"] = state["tool_calls"]
    if state["thinking_blocks"]:
        out["reasoning_state"] = _reasoning_state(state["thinking_blocks"])

    # Anthropic's input_tokens excludes cache reads and writes; the usage
    # counts every input token, cached ones included (the OpenAI convention).
    cache_read = state.get("cache_read_input_tokens", 0) or 0
    cache_write = state.get("cache_creation_input_tokens", 0) or 0
    prompt_tokens = (state.get("input_tokens", 0) or 0) + cache_read + cache_write
    output_tokens = state.get("output_tokens", 0) or 0
    if prompt_tokens or output_tokens:
        usage = LLMUsage(
            stop_reason=platform_stop_reason(state.get("stop_reason")),
            prompt_tokens=prompt_tokens,
            cached_tokens=cache_read,
            cache_write_tokens=cache_write,
            completion_tokens=output_tokens,
            total_tokens=prompt_tokens + output_tokens,
        )
        out["usage"] = usage
        out["output_meta"] = OutputMeta.from_llm_usage(usage)

    return out


def anthropic_tool_choice(choice: Optional[ToolChoice], model: str = "") -> Optional[Dict[str, Any]]:
    """LLMInput.tool_choice -> Anthropic tool_choice. None when the API default (auto) applies."""
    if choice is None or choice.mode == ToolChoiceMode.AUTO:
        return None
    if choice.mode == ToolChoiceMode.NONE:
        return {"type": "none"}
    if model in NO_FORCED_TOOL_CHOICE_MODELS:
        raise ValueError(
            f"{model} does not support tool_choice mode '{choice.mode.value}' (forced tool use); "
            "only 'auto' and 'none' are accepted. Ask for the tool in the prompt instead."
        )
    if choice.mode == ToolChoiceMode.REQUIRED:
        return {"type": "any"}
    if not choice.name:
        raise ValueError("tool_choice mode 'function' requires a tool name")
    return {"type": "tool", "name": choice.name}


def anthropic_output_format(fmt: Optional[ResponseFormat]) -> Optional[Dict[str, Any]]:
    """LLMInput.response_format -> Anthropic output_config.format.

    Anthropic's structured output is schema-driven; a bare json_object mode
    has no native equivalent and is rejected rather than approximated.
    """
    if fmt is None or fmt.type == ResponseFormatType.TEXT:
        return None
    if fmt.type == ResponseFormatType.JSON_OBJECT:
        raise ValueError("response_format json_object is not supported by Anthropic; use json_schema with a schema")
    if not fmt.json_schema:
        raise ValueError("response_format json_schema requires json_schema")
    return {"type": "json_schema", "schema": fmt.json_schema}


def build_params(
    input_data,
    model: str,
    max_tokens: int = 64000,
) -> Dict[str, Any]:
    """Build common request parameters for Anthropic API."""
    system_prompt, messages = convert_messages_to_anthropic(input_data, model)
    tools = convert_tools_to_anthropic(input_data.tools) if input_data.tools else None
    tool_choice = anthropic_tool_choice(input_data.tool_choice, model=model)
    output_format = anthropic_output_format(input_data.response_format)

    # The caller's max_tokens (LLMInput defaults it to 64000), capped at the
    # model's limit; thinking shares the budget.
    requested = getattr(input_data, "max_tokens", None) or max_tokens
    params: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max(1, min(int(requested), max_tokens)),
        "stream": True,
    }

    if system_prompt:
        params["system"] = system_prompt

    if tools:
        params["tools"] = tools
        if tool_choice is not None:
            params["tool_choice"] = tool_choice
    elif tool_choice is not None and tool_choice["type"] != "none":
        raise ValueError("tool_choice requires tools")

    thinking = build_thinking_param(input_data, model=model)
    if thinking is not None:
        params["thinking"] = thinking

    output_config = build_output_config(input_data, model=model) or {}
    if output_format is not None:
        output_config["format"] = output_format
    if output_config:
        params["output_config"] = output_config

    params["cache_control"] = {"type": "ephemeral"}

    return params


def apply_stream_event(event, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Fold one Anthropic stream event into state.

    Returns the LLMDelta-shaped increment this event produced (response,
    reasoning, tool_calls with index/id/type/function fragments), or None.
    """
    delta: Dict[str, Any] = {}

    if event.type == "content_block_start":
        block = event.content_block
        if block.type == "thinking":
            state["current_thinking"] = {
                "type": "thinking",
                "thinking": getattr(block, "thinking", "") or "",
                "signature": getattr(block, "signature", "") or "",
            }
            state["thinking_blocks"].append(state["current_thinking"])
        elif block.type == "redacted_thinking":
            state["thinking_blocks"].append({"type": "redacted_thinking", "data": block.data})
        elif block.type == "tool_use":
            state["current_tool"] = {
                "id": block.id,
                "type": "function",
                "function": {"name": block.name, "arguments": ""},
            }
            state["tool_calls"].append(state["current_tool"])
            delta["tool_calls"] = [{
                "index": len(state["tool_calls"]) - 1,
                "id": block.id,
                "type": "function",
                "function": {"name": block.name, "arguments": ""},
            }]

    elif event.type == "content_block_delta":
        d = event.delta
        if d.type == "text_delta" and d.text:
            state["response"] += d.text
            delta["response"] = d.text
        elif d.type == "thinking_delta" and d.thinking:
            state["thinking"] += d.thinking
            delta["reasoning"] = d.thinking
            if state["current_thinking"] is not None:
                state["current_thinking"]["thinking"] += d.thinking
        elif d.type == "signature_delta" and state["current_thinking"] is not None:
            state["current_thinking"]["signature"] += d.signature
        elif d.type == "input_json_delta" and state["current_tool"] and d.partial_json:
            state["current_tool"]["function"]["arguments"] += d.partial_json
            delta["tool_calls"] = [{
                "index": len(state["tool_calls"]) - 1,
                "function": {"arguments": d.partial_json},
            }]

    elif event.type == "content_block_stop":
        state["current_tool"] = None
        state["current_thinking"] = None

    elif event.type == "message_delta":
        usage = getattr(event, "usage", None)
        if usage:
            state["output_tokens"] = getattr(usage, "output_tokens", 0)
        stop = getattr(getattr(event, "delta", None), "stop_reason", None)
        if stop:
            state["stop_reason"] = stop

    elif event.type == "message_start":
        msg = getattr(event, "message", None)
        if msg:
            usage = getattr(msg, "usage", None)
            if usage:
                state["input_tokens"] = getattr(usage, "input_tokens", 0)
                state["cache_read_input_tokens"] = getattr(usage, "cache_read_input_tokens", 0)
                state["cache_creation_input_tokens"] = getattr(usage, "cache_creation_input_tokens", 0)

    return delta or None


async def stream_completion(
    client, input_data, model: str, max_tokens: int = 64000, *, with_deltas: bool = False,
) -> AsyncGenerator[Any, None]:
    """Stream a completion from Anthropic API.

    with_deltas=False (default): yields accumulated output dicts.
    with_deltas=True: yields (output_dict, delta_dict | None) tuples; delta_dict
    has LLMDelta-shaped keys.
    """
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

            delta = apply_stream_event(event, state)
            output = build_output(state)
            yield (output, delta) if with_deltas else output

    except Exception as e:
        if "overloaded" in str(e).lower():
            raise RuntimeError("Anthropic API is overloaded, please try again later")
        raise

    logger.info(
        f"Anthropic stream complete: in={state['input_tokens']} out={state['output_tokens']}"
        f" cache_read={state.get('cache_read_input_tokens', 0)}"
        f" cache_write={state.get('cache_creation_input_tokens', 0)}"
    )


async def complete(
    client, input_data, model: str, max_tokens: int = 64000
) -> Dict[str, Any]:
    """Non-streaming completion from Anthropic API.

    max_tokens is capped for this path: the SDK refuses a non-streaming request
    whose estimated duration exceeds the client timeout, and a response that big
    would not finish inside NON_STREAM_TIMEOUT anyway. Use stream=True for long
    outputs.
    """
    params = build_params(input_data, model, min(max_tokens, NON_STREAM_MAX_TOKENS))
    params["stream"] = False

    logger.info(f"Calling Anthropic API (non-stream): model={model}")

    try:
        # The explicit timeout also suppresses the SDK's long-request guard.
        response = await asyncio.wait_for(
            client.with_options(timeout=NON_STREAM_TIMEOUT).messages.create(**params),
            timeout=NON_STREAM_TIMEOUT,
        )
    except asyncio.TimeoutError:
        raise RuntimeError("Anthropic API call timed out after 120 seconds")
    except Exception as e:
        raise handle_api_error(e)

    state = create_initial_state()

    state["stop_reason"] = getattr(response, "stop_reason", None) or ""

    # Extract usage
    if hasattr(response, "usage") and response.usage:
        state["input_tokens"] = getattr(response.usage, "input_tokens", 0)
        state["output_tokens"] = getattr(response.usage, "output_tokens", 0)
        state["cache_read_input_tokens"] = getattr(response.usage, "cache_read_input_tokens", 0)
        state["cache_creation_input_tokens"] = getattr(response.usage, "cache_creation_input_tokens", 0)

    # Process content blocks
    for block in response.content:
        if block.type == "text":
            state["response"] += block.text
        elif block.type == "thinking":
            state["thinking"] += block.thinking
            state["thinking_blocks"].append(
                {"type": "thinking", "thinking": block.thinking, "signature": block.signature}
            )
        elif block.type == "redacted_thinking":
            state["thinking_blocks"].append({"type": "redacted_thinking", "data": block.data})
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
        f" cache_read={state.get('cache_read_input_tokens', 0)}"
        f" cache_write={state.get('cache_creation_input_tokens', 0)}"
    )

    return build_output(state)

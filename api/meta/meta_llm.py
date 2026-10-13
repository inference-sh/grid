"""Meta Model API Responses streaming helper — raw httpx SSE, no SDK.

Shared by the meta/muse-* chat apps (symlinked into each app directory).
Adapted from openai/openai_llm.py; Meta's Responses API follows OpenAI's wire
format (dev.meta.ai/docs/protocols/responses) with these differences, all
handled here:

- Reasoning cannot be turned off: effort "none" is HTTP 400. The accepted set
  is per model (muse-wonder-1.0: minimal..max). A requested value outside the
  set maps to the nearest one, so the platform-wide default "none" becomes
  "minimal".
- Raw reasoning is never streamed as text; it comes back encrypted for replay
  (include: reasoning.encrypted_content, store: false) and as a summary when
  reasoning.summary is requested. The summary is what `reasoning` carries.
- A replayed reasoning item must carry `summary` (an empty list when none).
- tool_choice: only "auto" works. "none" is honoured by not sending tools;
  "required" and a named function are refused before the request.
- Images are processed only in user messages and function_call_output is a
  string. A tool result that returned images or files sends its text as the
  output and the media in a user message right after it.
- Meta fetches image_url and file_url itself and refuses an origin whose
  robots.txt disallows it (400 media_url_robots_disallowed). The platform has
  already downloaded every input, so the local copy goes inline as a data URL
  and a URL is passed on only when there is no local copy.
- input_image has no `detail` parameter. JPEG, PNG, GIF, WebP and ICO up to
  50 MB each, 50 per request. Another format is converted from the local copy.
- Files go as input_file with file_data as a data URL (file_url when there is
  no local copy). Documented types: pdf, mp4, mp3, wav, txt, json.
- temperature, top_p, frequency_penalty and presence_penalty are documented
  and sent when the caller set them (not the LLMInput defaults). top_k, seed
  and stop are not supported and never sent.
- max_output_tokens is the caller's max_tokens (platform default 64000),
  capped at the model's limit: an over-limit value is HTTP 400, and reasoning
  shares the budget. Nothing is sent when max_tokens is unset, so the model
  budgets reasoning plus output itself.
- A stream-level `error` event with code server_shutting_down before any
  output means "retry the whole request", and is retried here.
- Meta sheds load on its newest model with HTTP 404 model_not_found instead
  of a 503: the same key, worker, connection and request flip between 200 and
  404 within seconds, the 404 comes back from the router in ~400 ms, and
  /v1/models keeps listing the model. Seen on muse-spark-1.3 in September
  (CLIProxyAPI #6117, popcre/ai-devops #743) and on muse-wonder-1.0 the day
  after its launch. That response is a 503 service_overloaded here: retried
  and reported as one.
"""

import asyncio
import base64
import json
import mimetypes
import os
import time
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import httpx

from inferencesh.images import ImageLimits, fit_image
from inferencesh import OutputMeta, TextMeta
from inferencesh.llm_types_gen import ResponseFormat, ResponseFormatType, ToolChoice, ToolChoiceMode
from inferencesh.models.llm import (
    ContextMessageRole,
    LLMUsage,
    build_tools as _sdk_build_tools,
)

META_BASE_URL = "https://api.meta.ai/v1"

# Connection-level retries only. There is deliberately no read timeout on the
# stream: at high reasoning effort the API can be silent for many minutes before
# the first event, and the platform owns task-level timeouts.
MAX_CONNECT_RETRIES = 3
# 429 and 503 carry Retry-After; 500 is "retry with backoff"; 504 is the
# non-streaming time limit and is not expected on a streamed request.
RETRYABLE_STATUS = {408, 429, 500, 502, 503}

# Reasoning summaries are documented, but the parameter is dropped for the life
# of the worker if the API ever rejects it, as the OpenAI helper does.
_summary_supported = True


def get_api_key() -> str:
    key = os.environ.get("META_AI_KEY")
    if not key:
        raise RuntimeError(
            "META_AI_KEY is not set. A secret whose record exists but holds an empty value is "
            "not injected at all: check that `belt secrets get META_AI_KEY --json` reports a "
            "non-empty masked_value, and re-set it if it does not."
        )
    return key.strip()


# ---------------------------------------------------------------------------
# Input conversion — LLMInput -> Responses API `input` items
# ---------------------------------------------------------------------------

# The key this helper's entry has in an output's reasoning_state, the opaque
# state the platform stores with the assistant message and sends back on every
# later turn. The entry is {"items": [reasoning output items]}: Meta's
# reasoning, encrypted (we run with store: false, so nothing is kept server
# side). Only Meta can decrypt it, so another helper's entry is never read.
REASONING_STATE_KEY = "meta"


def _reasoning_items(msg) -> List[Dict[str, Any]]:
    """The reasoning items this helper stored on an assistant message, in the
    shape the next request takes them back: `summary` is required on input."""
    entry = (getattr(msg, "reasoning_state", None) or {}).get(REASONING_STATE_KEY)
    if not isinstance(entry, dict):
        return []
    items = []
    for item in entry.get("items") or []:
        if isinstance(item, dict) and item.get("type") == "reasoning":
            items.append({**item, "summary": item.get("summary") or []})
    return items


def _is_http_url(s: Optional[str]) -> bool:
    return bool(s) and (s.startswith("http://") or s.startswith("https://"))


# Meta takes JPEG, PNG, GIF, WebP and ICO up to 50 MB inline, 50 per request
# (dev.meta.ai/docs/image-understanding). Another format is converted from the
# local copy.
IMAGE_LIMITS = ImageLimits(
    max_bytes=50 * 1000 * 1000,
    mime_types=("image/png", "image/jpeg", "image/gif", "image/webp", "image/x-icon"),
)
MAX_IMAGES_PER_REQUEST = 50


def _image_part(image) -> Optional[Dict[str, Any]]:
    if image is None:
        return None
    if image.path:
        data, mime = fit_image(image.path, IMAGE_LIMITS)
        url = f"data:{mime};base64,{base64.b64encode(data).decode()}"
    elif image.uri:
        url = image.uri
    else:
        return None
    return {"type": "input_image", "image_url": url}


def _file_data_uri(path: str, filename: Optional[str]) -> str:
    mime = mimetypes.guess_type(filename or path)[0] or "application/octet-stream"
    with open(path, "rb") as f:
        return f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"


def _file_part(file) -> Optional[Dict[str, Any]]:
    if file is None:
        return None
    filename = getattr(file, "filename", None) or "file"
    if file.path:
        return {"type": "input_file", "filename": filename, "file_data": _file_data_uri(file.path, filename)}
    if _is_http_url(file.uri):
        return {"type": "input_file", "file_url": file.uri}
    if file.uri:
        return {"type": "input_file", "filename": filename, "file_data": file.uri}
    return None


def _user_content(text: Optional[str], images, files) -> Any:
    parts: List[Dict[str, Any]] = []
    if text:
        parts.append({"type": "input_text", "text": text})
    for image in images or []:
        part = _image_part(image)
        if part:
            parts.append(part)
    for file in files or []:
        part = _file_part(file)
        if part:
            parts.append(part)
    if not parts:
        return text or ""
    if len(parts) == 1 and parts[0]["type"] == "input_text":
        return parts[0]["text"]
    return parts


def _function_call_output_items(call_id: str, text: Optional[str], images, files) -> List[Dict[str, Any]]:
    """A tool's result. The output is a string; Meta reads images only in user
    messages, so a result that returned images or files is followed by a user
    message carrying them."""
    items: List[Dict[str, Any]] = [{"type": "function_call_output", "call_id": call_id, "output": text or ""}]
    if images or files:
        content = _user_content(f"Attachments returned by tool call {call_id}", images, files)
        if isinstance(content, list):
            items.append({"role": "user", "content": content})
    return items


def _function_call_items(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Assistant tool calls -> function_call items.

    The item `id` (fc_...) is intentionally omitted: with it present the API
    demands the matching reasoning item, which a turn written by another app
    (or before reasoning state was kept) does not have. `call_id` is all that
    is needed to pair with a function_call_output.
    """
    items = []
    for tc in tool_calls:
        fn = tc.get("function", {}) or {}
        args = fn.get("arguments", "")
        if not isinstance(args, str):
            args = json.dumps(args)
        items.append({
            "type": "function_call",
            "call_id": tc.get("id", ""),
            "name": fn.get("name", ""),
            "arguments": args or "{}",
        })
    return items


def build_input(input_data) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """Convert LLMInput (context + current turn) to (instructions, input items)."""
    instructions = input_data.system_prompt or None
    items: List[Dict[str, Any]] = []

    for msg in input_data.context:
        role = msg.role

        if role == ContextMessageRole.TOOL and msg.tool_call_id:
            items.extend(_function_call_output_items(
                msg.tool_call_id, msg.text, getattr(msg, "images", None), getattr(msg, "files", None)))
            continue

        if role == ContextMessageRole.ASSISTANT:
            # The turn's reasoning goes back before what it produced, as the
            # model emitted it.
            items.extend(_reasoning_items(msg))
            if msg.text:
                items.append({
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": msg.text}],
                })
            if msg.tool_calls:
                items.extend(_function_call_items(msg.tool_calls))
            continue

        if role == ContextMessageRole.USER:
            content = _user_content(msg.text, getattr(msg, "images", None), getattr(msg, "files", None))
            if content:
                items.append({"role": "user", "content": content})
            continue

        # SYSTEM / INJECTION / COMPACTION: additional system guidance mid-context.
        if msg.text:
            items.append({"role": "system", "content": msg.text})

    # Current turn
    if input_data.role == ContextMessageRole.TOOL and input_data.tool_call_id:
        items.extend(_function_call_output_items(
            input_data.tool_call_id, input_data.text, input_data.images, input_data.files))
    else:
        files = list(input_data.files or []) + list(getattr(input_data, "attachments", None) or [])
        content = _user_content(input_data.text, input_data.images, files)
        if content:
            items.append({"role": "user", "content": content})

    images = sum(
        1 for item in items
        if isinstance(item.get("content"), list)
        for part in item["content"] if part.get("type") == "input_image"
    )
    if images > MAX_IMAGES_PER_REQUEST:
        raise ValueError(f"Meta takes at most {MAX_IMAGES_PER_REQUEST} images per request; this one has {images}")

    return instructions, items


def build_tools(tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    """OpenAI chat-completions style tools -> Responses API flat function tools."""
    wrapped = _sdk_build_tools(tools)
    if not wrapped:
        return None
    result = []
    for tool in wrapped:
        fn = tool.get("function", tool)
        result.append({
            "type": "function",
            "name": fn.get("name", ""),
            "description": fn.get("description", "") or "",
            "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
        })
    return result


def _effort_value(input_data) -> Optional[str]:
    effort = input_data.reasoning_effort
    if isinstance(effort, Enum):
        effort = effort.value
    return effort or None


_EFFORT_ORDER = ["none", "minimal", "low", "medium", "high", "xhigh", "max"]


def build_reasoning(input_data, efforts: Optional[Tuple[str, ...]], with_summary: bool) -> Optional[Dict[str, Any]]:
    """Build the `reasoning` request parameter.

    `efforts` is the model's accepted set, lowest first; None means the model
    publishes no effort control and no effort is sent. A value outside the set
    maps to the nearest accepted one, so the platform-wide default "none"
    becomes the cheapest valid request (muse-wonder-1.0: "minimal").
    """
    cfg: Dict[str, Any] = {}
    effort = _effort_value(input_data)
    if effort and efforts:
        if effort not in efforts:
            rank = _EFFORT_ORDER.index(effort) if effort in _EFFORT_ORDER else 0
            effort = min(efforts, key=lambda e: abs(_EFFORT_ORDER.index(e) - rank))
        cfg["effort"] = effort
    if with_summary:
        cfg["summary"] = "auto"
    return cfg or None


def responses_tool_choice(choice: Optional[ToolChoice]) -> str:
    """LLMInput.tool_choice -> Responses API tool_choice.

    Meta accepts only "auto"; "none" is handled by the caller (tools are not
    sent) and the rest are refused here rather than sent for an HTTP 400.
    """
    if choice is None or choice.mode == ToolChoiceMode.AUTO:
        return "auto"
    raise ValueError(f"Meta does not support tool_choice mode '{choice.mode.value}'; only 'auto' and 'none' are accepted")


def responses_text_format(fmt: Optional[ResponseFormat]) -> Optional[Dict[str, Any]]:
    """LLMInput.response_format -> Responses API text.format. None for plain text.

    The Responses API flattens the schema spec: {"type": "json_schema",
    "name", "schema", "strict"} directly, not nested under "json_schema".
    """
    if fmt is None or fmt.type == ResponseFormatType.TEXT:
        return None
    if fmt.type == ResponseFormatType.JSON_OBJECT:
        return {"type": "json_object"}
    spec: Dict[str, Any] = {"type": "json_schema", "name": fmt.name or "response", "schema": fmt.json_schema}
    if fmt.strict is not None:
        spec["strict"] = fmt.strict
    return spec


def build_request_body(
    input_data,
    model: str,
    *,
    max_output_tokens: int,
    efforts: Optional[Tuple[str, ...]],
    with_summary: bool = True,
) -> Dict[str, Any]:
    instructions, items = build_input(input_data)
    choice = input_data.tool_choice
    tools_off = choice is not None and choice.mode == ToolChoiceMode.NONE
    tools = build_tools(input_data.tools) if input_data.tools and not tools_off else None

    body: Dict[str, Any] = {
        "model": model,
        "input": items,
        "stream": True,
        "store": False,
    }
    # Capped at the model limit (above it is HTTP 400); unset means the model
    # budgets reasoning plus output itself.
    if input_data.max_tokens:
        body["max_output_tokens"] = max(16, min(int(input_data.max_tokens), max_output_tokens))

    if instructions:
        body["instructions"] = instructions
    if tools:
        body["tools"] = tools
        body["tool_choice"] = responses_tool_choice(choice)
        body["parallel_tool_calls"] = True

    # Sampling: documented on the Responses API and taken by reasoning models.
    # Only what the caller set goes: LLMInput defaults temperature and top_p
    # (0.7, 0.95), and the model's own defaults are the better ones.
    # top_k, seed and stop are not supported (stop is accepted and ignored).
    for name in ("temperature", "top_p", "frequency_penalty", "presence_penalty"):
        value = getattr(input_data, name, None)
        if value is not None and name in input_data.model_fields_set:
            body[name] = value

    text_format = responses_text_format(input_data.response_format)
    if text_format is not None:
        body["text"] = {"format": text_format}

    reasoning = build_reasoning(input_data, efforts, with_summary)
    if reasoning:
        body["reasoning"] = reasoning
    # With store: false the reasoning is kept only if it comes back encrypted,
    # for reasoning_state to carry to the next turn.
    body["include"] = ["reasoning.encrypted_content"]

    return body


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

def normalize_capacity_rejection(status: int, body: Any) -> Tuple[int, Any]:
    """A 404 model_not_found is Meta's capacity rejection (module docstring):
    returned as the 503 it is, so retries and the error read it as one."""
    if status != 404 or not isinstance(body, dict):
        return status, body
    err = body.get("error") or {}
    if (err.get("code") or "") != "model_not_found":
        return status, body
    return 503, {"error": {
        "type": "server_error",
        "code": "service_overloaded",
        "param": err.get("param"),
        "message": "Meta has no capacity for this model right now (reported upstream as 404 model_not_found)",
    }}


def _format_error(status: int, body: Any, headers: Dict[str, str], prefix: str = "Meta API") -> str:
    request_id = headers.get("x-request-id", "")
    err = body.get("error", {}) if isinstance(body, dict) else {}
    if isinstance(err, str):
        err = {"message": err}
    msg = err.get("message") or (body if isinstance(body, str) else f"HTTP {status}")
    detail = " ".join(str(p) for p in (status, err.get("type"), err.get("code"), err.get("param")) if p)
    text = f"{prefix} error ({detail}): {msg}"
    if request_id:
        text += f" [req:{request_id}]"
    return text


def _is_summary_rejection(status: int, body: Any) -> bool:
    if status != 400 or not isinstance(body, dict):
        return False
    err = body.get("error", {}) or {}
    blob = f"{err.get('param', '')} {err.get('message', '')}".lower()
    return "summary" in blob


class _RetryStream(Exception):
    """The stream ended before any output with a failure Meta says to retry."""


# ---------------------------------------------------------------------------
# SSE event handling
# ---------------------------------------------------------------------------

def _create_initial_state() -> Dict[str, Any]:
    return {
        "response": "",
        "reasoning": "",
        "tool_calls": [],
        "_item_index": {},        # response item id -> index in tool_calls
        "_summary_parts": 0,
        "reasoning_items": [],    # reasoning output items, in order, for reasoning_state
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
        "response_id": None,
        "status": None,
        "incomplete_reason": None,
        "started_at": time.monotonic(),
        "first_token_at": None,
    }


def _has_output(state: Dict[str, Any]) -> bool:
    return bool(state["response"] or state["reasoning"] or state["tool_calls"])


def _handle_event(data: Dict[str, Any], state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Fold one Responses API stream event into state.
    Returns an LLMDelta-shaped dict when the event produced output, else None."""
    etype = data.get("type", "")

    if etype == "response.output_text.delta":
        text = data.get("delta") or ""
        if not text:
            return None
        if state["first_token_at"] is None:
            state["first_token_at"] = time.monotonic()
        state["response"] += text
        return {"response": text}

    if etype == "response.reasoning_summary_part.added":
        # Separate consecutive summary paragraphs so the concatenated reasoning reads well.
        state["_summary_parts"] += 1
        if state["reasoning"] and not state["reasoning"].endswith("\n"):
            state["reasoning"] += "\n\n"
            return {"reasoning": "\n\n"}
        return None

    if etype == "response.reasoning_summary_text.delta":
        text = data.get("delta") or ""
        if not text:
            return None
        if state["first_token_at"] is None:
            state["first_token_at"] = time.monotonic()
        state["reasoning"] += text
        return {"reasoning": text}

    if etype == "response.output_item.added":
        item = data.get("item") or {}
        if item.get("type") == "function_call":
            idx = len(state["tool_calls"])
            call_id = item.get("call_id") or item.get("id") or ""
            name = item.get("name") or ""
            args = item.get("arguments") or ""
            state["tool_calls"].append({
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": args},
            })
            if item.get("id"):
                state["_item_index"][item["id"]] = idx
            return {"tool_calls": [{
                "index": idx,
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": args},
            }]}
        return None

    if etype == "response.function_call_arguments.delta":
        fragment = data.get("delta") or ""
        idx = _call_index(state, data)
        if idx is None or not fragment:
            return None
        state["tool_calls"][idx]["function"]["arguments"] += fragment
        return {"tool_calls": [{"index": idx, "function": {"arguments": fragment}}]}

    if etype == "response.function_call_arguments.done":
        idx = _call_index(state, data)
        final_args = data.get("arguments")
        if idx is None or final_args is None:
            return None
        return _settle_arguments(state, idx, final_args)

    if etype == "response.output_item.done":
        item = data.get("item") or {}
        if item.get("type") == "reasoning":
            state["reasoning_items"].append(_replayable_reasoning_item(item))
            return None
        if item.get("type") != "function_call":
            return None
        idx = state["_item_index"].get(item.get("id"))
        if idx is None and item.get("call_id"):
            idx = next((i for i, tc in enumerate(state["tool_calls"]) if tc["id"] == item["call_id"]), None)
        if idx is None:
            # No matching .added event: the call arrives only here.
            return _handle_event({"type": "response.output_item.added", "item": item}, state)
        if item.get("call_id"):
            state["tool_calls"][idx]["id"] = item["call_id"]
        final_args = item.get("arguments")
        if final_args:
            return _settle_arguments(state, idx, final_args)
        return None

    if etype in ("response.completed", "response.incomplete"):
        resp = data.get("response") or {}
        state["response_id"] = resp.get("id") or state["response_id"]
        state["status"] = resp.get("status") or etype.split(".")[-1]
        if etype == "response.incomplete":
            state["incomplete_reason"] = (resp.get("incomplete_details") or {}).get("reason") or "incomplete"
        usage = resp.get("usage") or {}
        if usage:
            state["input_tokens"] = usage.get("input_tokens") or 0
            state["output_tokens"] = usage.get("output_tokens") or 0
            state["cached_tokens"] = (usage.get("input_tokens_details") or {}).get("cached_tokens") or 0
            state["reasoning_tokens"] = (usage.get("output_tokens_details") or {}).get("reasoning_tokens") or 0
            state["total_tokens"] = usage.get("total_tokens") or 0
        return None

    if etype == "response.created":
        resp = data.get("response") or {}
        state["response_id"] = resp.get("id")
        return None

    if etype == "response.failed":
        err = ((data.get("response") or {}).get("error") or {})
        raise RuntimeError(f"Meta response failed: [{err.get('code', '')}] {err.get('message', 'unknown error')}")

    if etype == "error":
        err = data.get("error") if isinstance(data.get("error"), dict) else data
        code = err.get("code", "") or ""
        if code == "server_shutting_down" and not _has_output(state):
            raise _RetryStream(code)
        raise RuntimeError(f"Meta stream error: [{code}] {err.get('message', 'unknown error')}")

    return None


def _call_index(state: Dict[str, Any], data: Dict[str, Any]) -> Optional[int]:
    """The tool call an arguments event belongs to: by item id, else by
    output_index order when the call carried no id."""
    idx = state["_item_index"].get(data.get("item_id"))
    if idx is None and not state["_item_index"] and state["tool_calls"]:
        idx = len(state["tool_calls"]) - 1
    return idx


def _settle_arguments(state: Dict[str, Any], idx: int, final_args: str) -> Optional[Dict[str, Any]]:
    """Make the accumulated arguments match the authoritative string, emitting
    the missing tail when deltas were incomplete."""
    current = state["tool_calls"][idx]["function"]["arguments"]
    if final_args == current:
        return None
    state["tool_calls"][idx]["function"]["arguments"] = final_args
    if final_args.startswith(current) and final_args[len(current):]:
        return {"tool_calls": [{"index": idx, "function": {"arguments": final_args[len(current):]}}]}
    return None


# ---------------------------------------------------------------------------
# Output building
# ---------------------------------------------------------------------------

def _stop_reason(state: Dict[str, Any]) -> str:
    if state.get("incomplete_reason") == "max_output_tokens":
        return "length"
    if state.get("incomplete_reason"):
        return state["incomplete_reason"]
    if state["tool_calls"]:
        return "tool_calls"
    return "stop" if state.get("status") else ""


def _replayable_reasoning_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """A reasoning output item as the next request takes it back: everything
    but `status`, which is output-only, and with `summary` always present."""
    out = {k: v for k, v in item.items() if k != "status"}
    out["summary"] = out.get("summary") or []
    return out


def billable_output_tokens(state: Dict[str, Any]) -> int:
    """Output tokens including reasoning.

    Meta documents reasoning tokens as a detail of output_tokens; total_tokens
    settles it per response in case they are reported separately.
    """
    out, reasoning = state["output_tokens"], state["reasoning_tokens"]
    if reasoning and state["total_tokens"] and state["total_tokens"] - state["input_tokens"] == out + reasoning:
        return out + reasoning
    return out


def _build_output(state: Dict[str, Any], final: bool = False) -> Dict[str, Any]:
    out: Dict[str, Any] = {"response": state["response"]}
    if state["reasoning"]:
        out["reasoning"] = state["reasoning"]
    if state["tool_calls"]:
        out["tool_calls"] = state["tool_calls"]

    if not final:
        return out

    if state["reasoning_items"]:
        out["reasoning_state"] = {REASONING_STATE_KEY: {"items": state["reasoning_items"]}}

    output_tokens = billable_output_tokens(state)
    elapsed = max(time.monotonic() - state["started_at"], 1e-6)
    ttft = (state["first_token_at"] - state["started_at"]) if state["first_token_at"] else 0.0
    gen_time = max(elapsed - ttft, 1e-6)
    out["usage"] = LLMUsage(
        stop_reason=_stop_reason(state),
        time_to_first_token=round(ttft, 3),
        tokens_per_second=round(output_tokens / gen_time, 2) if output_tokens else 0.0,
        prompt_tokens=state["input_tokens"],
        cached_tokens=state["cached_tokens"],
        completion_tokens=output_tokens,
        total_tokens=state["input_tokens"] + output_tokens,
        reasoning_tokens=state["reasoning_tokens"],
    )
    meta = OutputMeta.from_llm_usage(out["usage"])
    if meta.inputs or meta.outputs:
        out["output_meta"] = meta
    return out


# ---------------------------------------------------------------------------
# Main streaming function
# ---------------------------------------------------------------------------

async def stream_completion(
    input_data,
    model: str,
    *,
    max_output_tokens: int = 131072,
    efforts: Optional[Tuple[str, ...]] = None,
    with_deltas: bool = False,
) -> AsyncGenerator[Any, None]:
    """Stream a Responses API completion.

    with_deltas=False: yields accumulated output dicts (last one carries usage + output_meta).
    with_deltas=True:  yields (output_dict, delta_dict | None) tuples; delta_dict has
                       LLMDelta-shaped keys (response, reasoning, tool_calls).
    """
    global _summary_supported

    headers = {
        "Authorization": f"Bearer {get_api_key()}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    timeout = httpx.Timeout(connect=15.0, read=None, write=30.0, pool=15.0)
    state = _create_initial_state()
    events = 0

    async with httpx.AsyncClient(timeout=timeout) as http:
        for attempt in range(1, MAX_CONNECT_RETRIES + 1):
            body = build_request_body(
                input_data, model,
                max_output_tokens=max_output_tokens,
                efforts=efforts,
                with_summary=_summary_supported,
            )
            print(
                f"Calling Meta model={model} attempt={attempt} items={len(body['input'])}"
                f" tools={len(body.get('tools') or [])} reasoning={body.get('reasoning')}"
                f" max_output_tokens={body.get('max_output_tokens')}"
            )
            try:
                req = http.build_request("POST", f"{META_BASE_URL}/responses", json=body, headers=headers)
                resp = await http.send(req, stream=True)
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.WriteTimeout, httpx.PoolTimeout) as e:
                if attempt < MAX_CONNECT_RETRIES:
                    print(f"  Connection failed ({type(e).__name__}) on attempt {attempt}, retrying...")
                    await asyncio.sleep(attempt)
                    continue
                raise RuntimeError(f"Meta API connection failed after {attempt} attempts: {e}")

            if resp.status_code != 200:
                raw = await resp.aread()
                await resp.aclose()
                try:
                    err_body = json.loads(raw)
                except json.JSONDecodeError:
                    err_body = raw.decode("utf-8", "replace")[:500]
                resp_headers = dict(resp.headers)
                if 400 <= resp.status_code < 500:
                    print(f"  HTTP {resp.status_code} body: {json.dumps(err_body)[:1000]}")
                status, err_body = normalize_capacity_rejection(resp.status_code, err_body)

                if _is_summary_rejection(status, err_body) and _summary_supported:
                    print("  Reasoning summaries rejected by the API; retrying without summary")
                    _summary_supported = False
                    continue

                if status in RETRYABLE_STATUS and attempt < MAX_CONNECT_RETRIES:
                    retry_after = resp_headers.get("retry-after")
                    backoff = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else 2.0 ** attempt
                    print(f"  HTTP {status} on attempt {attempt}, backing off {backoff:.0f}s")
                    await asyncio.sleep(min(backoff, 60.0))
                    continue

                raise RuntimeError(_format_error(status, err_body, resp_headers))

            request_id = resp.headers.get("x-request-id", "")
            print(f"Stream opened model={model} req={request_id or 'unknown'}")
            state = _create_initial_state()
            events = 0
            try:
                async for line in resp.aiter_lines():
                    if not line or line.startswith(":") or line.startswith("event:"):
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

                    events += 1
                    delta = _handle_event(data, state)
                    if delta is None:
                        continue
                    output = _build_output(state)
                    yield (output, delta) if with_deltas else output
            except _RetryStream as e:
                if attempt < MAX_CONNECT_RETRIES:
                    print(f"  Stream ended before output ({e}) on attempt {attempt}, retrying...")
                    await asyncio.sleep(attempt)
                    continue
                raise RuntimeError(f"Meta stream error: [{e}] after {attempt} attempts")
            finally:
                await resp.aclose()
            break

    if state["incomplete_reason"]:
        print(f"Response incomplete: reason={state['incomplete_reason']}")

    print(
        f"Stream done model={model} resp={state.get('response_id') or '?'} events={events}"
        f" status={state.get('status')} in={state['input_tokens']} cached={state['cached_tokens']}"
        f" out={state['output_tokens']} reasoning={state['reasoning_tokens']} total={state['total_tokens']}"
        f" billed_out={billable_output_tokens(state)} tool_calls={len(state['tool_calls'])}"
    )

    final = _build_output(state, final=True)
    yield (final, None) if with_deltas else final

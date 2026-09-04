"""OpenAI Responses API streaming helper — raw httpx SSE, no OpenAI SDK.

Shared by the openai/gpt-* chat apps (symlinked into each app directory).

Streams text, reasoning summaries and function calls from the Responses API
and yields (output_dict, delta_dict) pairs: `output_dict` is the accumulated
LLMOutput-shaped state, `delta_dict` is the LLMDelta-shaped increment for the
chunk that was just parsed (or None when the event carried nothing to emit).

Why raw SSE rather than the SDK: the delta contract needs the exact per-chunk
increments (text, reasoning summary, tool call argument fragments) plus the
usage details (cached tokens, reasoning tokens) that arrive on the final
`response.completed` event. Parsing the wire format directly keeps that in one
place and avoids SDK version churn.
"""

import asyncio
import json
import time
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import httpx

from inferencesh import OutputMeta, TextMeta
from inferencesh.llm_types_gen import ResponseFormat, ResponseFormatType, ToolChoice, ToolChoiceMode
from inferencesh.models.llm import (
    ContextMessageRole,
    LLMUsage,
    build_tools as _sdk_build_tools,
    file_to_base64_data_uri,
    image_to_base64_data_uri,
)

OPENAI_BASE_URL = "https://api.openai.com/v1"

# Connection-level retries only. There is deliberately no read timeout on the
# stream: at high reasoning effort the API can be silent for many minutes before
# the first event, and the platform owns task-level timeouts.
MAX_CONNECT_RETRIES = 3
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}

# Reasoning summaries need an org that has completed OpenAI's verification.
# When the API rejects the `summary` parameter we drop it for the lifetime of
# the worker instead of failing every request.
_summary_supported = True


# ---------------------------------------------------------------------------
# Input conversion — LLMInput -> Responses API `input` items
# ---------------------------------------------------------------------------

def _is_http_url(s: Optional[str]) -> bool:
    return bool(s) and (s.startswith("http://") or s.startswith("https://"))


def _image_part(image) -> Optional[Dict[str, Any]]:
    if image is None:
        return None
    if _is_http_url(image.uri):
        url = image.uri
    elif image.path:
        url = image_to_base64_data_uri(image.path)
    elif image.uri:
        url = image.uri
    else:
        return None
    return {"type": "input_image", "image_url": url, "detail": "auto"}


def _file_part(file) -> Optional[Dict[str, Any]]:
    if file is None:
        return None
    filename = getattr(file, "filename", None) or "file"
    if _is_http_url(file.uri):
        return {"type": "input_file", "file_url": file.uri}
    if file.path:
        return {"type": "input_file", "filename": filename, "file_data": file_to_base64_data_uri(file.path)}
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


def _function_call_items(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Assistant tool calls -> function_call items.

    The item `id` (fc_...) is intentionally omitted: with it present the API
    demands the matching reasoning item, which we do not persist across turns.
    `call_id` is all that is needed to pair with a function_call_output.
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
            items.append({
                "type": "function_call_output",
                "call_id": msg.tool_call_id,
                "output": msg.text or "",
            })
            continue

        if role == ContextMessageRole.ASSISTANT:
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
        items.append({
            "type": "function_call_output",
            "call_id": input_data.tool_call_id,
            "output": input_data.text or "",
        })
    else:
        files = list(input_data.files or []) + list(getattr(input_data, "attachments", None) or [])
        content = _user_content(input_data.text, input_data.images, files)
        if content:
            items.append({"role": "user", "content": content})

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
    effort = getattr(input_data, "reasoning_effort", None)
    ms = getattr(input_data, "model_settings", None)
    if ms is not None and getattr(ms, "reasoning_effort", None):
        effort = ms.reasoning_effort
    if isinstance(effort, Enum):
        effort = effort.value
    return effort or None


def build_reasoning(input_data, supports_none: bool, with_summary: bool) -> Optional[Dict[str, Any]]:
    """Build the `reasoning` request parameter.

    Models that do not accept effort "none" (GPT-6 Astra) get "low" instead so
    the platform-wide default still produces the cheapest valid request.
    """
    effort = _effort_value(input_data)
    if effort == "none" and not supports_none:
        effort = "low"
    cfg: Dict[str, Any] = {}
    if effort:
        cfg["effort"] = effort
    if with_summary and effort != "none":
        cfg["summary"] = "auto"
    return cfg or None


def responses_tool_choice(choice: Optional[ToolChoice]) -> Any:
    """LLMInput.tool_choice -> Responses API tool_choice.

    Same strings as Chat Completions; a named function is flat
    {"type": "function", "name": ...} rather than nested under "function".
    """
    if choice is None:
        return "auto"
    if choice.mode == ToolChoiceMode.FUNCTION:
        return {"type": "function", "name": choice.name}
    return choice.mode.value  # none | auto | required


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
    supports_none_reasoning: bool,
    with_summary: bool,
) -> Dict[str, Any]:
    instructions, items = build_input(input_data)
    tools = build_tools(input_data.tools) if input_data.tools else None

    # max_tokens: model_settings overrides the flat field; cap at the model limit.
    requested = getattr(input_data, "max_tokens", None) or max_output_tokens
    ms = getattr(input_data, "model_settings", None)
    if ms is not None and getattr(ms, "max_tokens", None):
        requested = ms.max_tokens

    body: Dict[str, Any] = {
        "model": model,
        "input": items,
        "stream": True,
        "store": False,
        "max_output_tokens": max(1, min(int(requested), max_output_tokens)),
    }
    # temperature / top_p are rejected by reasoning models, so they are never sent.

    if instructions:
        body["instructions"] = instructions
    if tools:
        body["tools"] = tools
        body["tool_choice"] = responses_tool_choice(input_data.tool_choice)
        body["parallel_tool_calls"] = True

    text_format = responses_text_format(input_data.response_format)
    if text_format is not None:
        body["text"] = {"format": text_format}

    reasoning = build_reasoning(input_data, supports_none_reasoning, with_summary)
    if reasoning:
        body["reasoning"] = reasoning

    return body


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

def _format_error(status: int, body: Any, headers: Dict[str, str], prefix: str = "OpenAI API") -> str:
    request_id = headers.get("x-request-id", "")
    err = body.get("error", {}) if isinstance(body, dict) else {}
    if isinstance(err, str):
        err = {"message": err}
    msg = err.get("message") or (body if isinstance(body, str) else f"HTTP {status}")
    detail = " ".join(str(p) for p in (status, err.get("type"), err.get("code")) if p)
    text = f"{prefix} error ({detail}): {msg}"
    if request_id:
        text += f" [req:{request_id}]"
    return text


def _is_summary_rejection(status: int, body: Any) -> bool:
    if status != 400 or not isinstance(body, dict):
        return False
    err = body.get("error", {}) or {}
    blob = f"{err.get('param', '')} {err.get('message', '')}".lower()
    return "summary" in blob or "verif" in blob


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
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
        "response_id": None,
        "status": None,
        "incomplete_reason": None,
        "started_at": time.monotonic(),
        "first_token_at": None,
    }


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
            state["tool_calls"].append({
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": item.get("arguments") or ""},
            })
            if item.get("id"):
                state["_item_index"][item["id"]] = idx
            return {"tool_calls": [{
                "index": idx,
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": ""},
            }]}
        return None

    if etype == "response.function_call_arguments.delta":
        fragment = data.get("delta") or ""
        idx = state["_item_index"].get(data.get("item_id"))
        if idx is None or not fragment:
            return None
        state["tool_calls"][idx]["function"]["arguments"] += fragment
        return {"tool_calls": [{"index": idx, "function": {"arguments": fragment}}]}

    if etype == "response.function_call_arguments.done":
        idx = state["_item_index"].get(data.get("item_id"))
        final_args = data.get("arguments")
        if idx is None or final_args is None:
            return None
        current = state["tool_calls"][idx]["function"]["arguments"]
        if final_args != current:
            # Deltas were incomplete; emit the missing tail so the merged delta
            # stream matches the authoritative arguments string.
            tail = final_args[len(current):] if final_args.startswith(current) else final_args
            state["tool_calls"][idx]["function"]["arguments"] = final_args
            if tail and final_args.startswith(current):
                return {"tool_calls": [{"index": idx, "function": {"arguments": tail}}]}
        return None

    if etype == "response.output_item.done":
        item = data.get("item") or {}
        if item.get("type") == "function_call" and item.get("id") in state["_item_index"]:
            idx = state["_item_index"][item["id"]]
            if item.get("arguments"):
                state["tool_calls"][idx]["function"]["arguments"] = item["arguments"]
            if item.get("call_id"):
                state["tool_calls"][idx]["id"] = item["call_id"]
        return None

    if etype in ("response.completed", "response.incomplete"):
        resp = data.get("response") or {}
        state["response_id"] = resp.get("id") or state["response_id"]
        state["status"] = resp.get("status") or etype.split(".")[-1]
        if etype == "response.incomplete":
            state["incomplete_reason"] = (resp.get("incomplete_details") or {}).get("reason") or "incomplete"
        usage = resp.get("usage") or {}
        if usage:
            state["input_tokens"] = usage.get("input_tokens", 0) or 0
            state["output_tokens"] = usage.get("output_tokens", 0) or 0
            state["cached_tokens"] = (usage.get("input_tokens_details") or {}).get("cached_tokens", 0) or 0
            state["reasoning_tokens"] = (usage.get("output_tokens_details") or {}).get("reasoning_tokens", 0) or 0
        return None

    if etype == "response.created":
        resp = data.get("response") or {}
        state["response_id"] = resp.get("id")
        return None

    if etype == "response.failed":
        err = ((data.get("response") or {}).get("error") or {})
        raise RuntimeError(f"OpenAI response failed: [{err.get('code', '')}] {err.get('message', 'unknown error')}")

    if etype == "error":
        err = data.get("error") if isinstance(data.get("error"), dict) else data
        raise RuntimeError(f"OpenAI stream error: [{err.get('code', '')}] {err.get('message', 'unknown error')}")

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


def _build_output(state: Dict[str, Any], final: bool = False) -> Dict[str, Any]:
    out: Dict[str, Any] = {"response": state["response"]}
    if state["reasoning"]:
        out["reasoning"] = state["reasoning"]
    if state["tool_calls"]:
        out["tool_calls"] = state["tool_calls"]

    if not final:
        return out

    inputs, outputs = [], []
    if state["input_tokens"]:
        inputs.append(TextMeta(tokens=state["input_tokens"], extra={
            "cache_read_tokens": state["cached_tokens"],
        }))
    if state["output_tokens"]:
        outputs.append(TextMeta(tokens=state["output_tokens"], extra={
            "reasoning_tokens": state["reasoning_tokens"],
        }))
    if inputs or outputs:
        out["output_meta"] = OutputMeta(inputs=inputs, outputs=outputs)

    elapsed = max(time.monotonic() - state["started_at"], 1e-6)
    ttft = (state["first_token_at"] - state["started_at"]) if state["first_token_at"] else 0.0
    gen_time = max(elapsed - ttft, 1e-6)
    out["usage"] = LLMUsage(
        stop_reason=_stop_reason(state),
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

async def stream_completion(
    api_key: str,
    input_data,
    model: str,
    *,
    max_output_tokens: int = 128000,
    supports_none_reasoning: bool = True,
    with_deltas: bool = False,
) -> AsyncGenerator[Any, None]:
    """Stream a Responses API completion.

    with_deltas=False: yields accumulated output dicts (last one carries usage + output_meta).
    with_deltas=True:  yields (output_dict, delta_dict | None) tuples; delta_dict has
                       LLMDelta-shaped keys (response, reasoning, tool_calls).
    """
    global _summary_supported

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    timeout = httpx.Timeout(connect=15.0, read=None, write=30.0, pool=15.0)

    async with httpx.AsyncClient(timeout=timeout) as http:
        resp = None
        for attempt in range(1, MAX_CONNECT_RETRIES + 1):
            body = build_request_body(
                input_data, model,
                max_output_tokens=max_output_tokens,
                supports_none_reasoning=supports_none_reasoning,
                with_summary=_summary_supported,
            )
            print(
                f"Calling OpenAI model={model} attempt={attempt} items={len(body['input'])}"
                f" tools={len(body.get('tools') or [])} reasoning={body.get('reasoning')}"
                f" max_output_tokens={body['max_output_tokens']}"
            )
            try:
                req = http.build_request("POST", f"{OPENAI_BASE_URL}/responses", json=body, headers=headers)
                resp = await http.send(req, stream=True)
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.WriteTimeout, httpx.PoolTimeout) as e:
                if attempt < MAX_CONNECT_RETRIES:
                    print(f"  Connection failed ({type(e).__name__}) on attempt {attempt}, retrying...")
                    await asyncio.sleep(attempt)
                    continue
                raise RuntimeError(f"OpenAI API connection failed after {attempt} attempts: {e}")

            if resp.status_code == 200:
                break

            raw = await resp.aread()
            await resp.aclose()
            try:
                err_body = json.loads(raw)
            except json.JSONDecodeError:
                err_body = raw.decode("utf-8", "replace")[:500]
            resp_headers = dict(resp.headers)

            if _is_summary_rejection(resp.status_code, err_body) and _summary_supported:
                print("  Reasoning summaries rejected by the API; retrying without summary")
                _summary_supported = False
                continue

            if resp.status_code in RETRYABLE_STATUS and attempt < MAX_CONNECT_RETRIES:
                retry_after = resp_headers.get("retry-after")
                backoff = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else 2.0 * attempt
                print(f"  HTTP {resp.status_code} on attempt {attempt}, backing off {backoff:.0f}s")
                await asyncio.sleep(min(backoff, 30.0))
                continue

            raise RuntimeError(_format_error(resp.status_code, err_body, resp_headers))

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
        finally:
            await resp.aclose()

    if state["incomplete_reason"]:
        print(f"Response incomplete: reason={state['incomplete_reason']}")

    print(
        f"Stream done model={model} resp={state.get('response_id') or '?'} events={events}"
        f" status={state.get('status')} in={state['input_tokens']} cached={state['cached_tokens']}"
        f" out={state['output_tokens']} reasoning={state['reasoning_tokens']}"
        f" tool_calls={len(state['tool_calls'])}"
    )

    final = _build_output(state, final=True)
    yield (final, None) if with_deltas else final

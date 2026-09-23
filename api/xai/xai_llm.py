"""xAI Responses API streaming helper — raw httpx SSE, no SDK.

Shared by the xai/grok-* chat apps (symlinked into each app directory).
Adapted from openai/openai_llm.py; xAI's Responses API follows OpenAI's wire
format with these differences, all handled here:

- Reasoning effort values differ per model (grok-4.7: low..xhigh, no "none";
  grok-4.3: none..xhigh). Requested values outside a model's set are mapped to
  the nearest one. `reasoning.summary` is ignored by xAI, so it is not sent.
- Reasoning text streams as `response.reasoning_text.delta` or
  `response.reasoning_summary_text.delta`; whichever arrives first is used.
- Function calls arrive whole in `response.output_item.added/.done` rather
  than as argument deltas.
- `usage.cost_in_usd_ticks` (1 tick = 1e-10 USD) is xAI's billed cost. It is
  stored as extra.upstream_cost_usd for comparison; pricing does not read it.
- xAI charges a $0.05 fee for requests its usage-guideline check rejects
  before generation. Such a rejection returns success with an empty response,
  a notice, and extra.usage_violation=1 so pricing can bill the fee.
- File input is not supported: xAI takes only file URLs/ids and turns every
  attachment into a paid document search.
"""

import asyncio
import json
import os
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
    image_to_base64_data_uri,
)

XAI_BASE_URL = "https://api.x.ai/v1"

# Connection-level retries only. There is deliberately no read timeout on the
# stream: at high reasoning effort the API can be silent for many minutes before
# the first event, and the platform owns task-level timeouts.
MAX_CONNECT_RETRIES = 3
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}

USD_PER_TICK = 1e-10

# xAI documents the violation fee but not the error it returns, so rejections
# are recognised by their message. Every 4xx body is logged to refine this.
_VIOLATION_MARKERS = ("usage guideline", "usage policy", "acceptable use", "content policy", "moderation")

VIOLATION_NOTICE = (
    "xAI rejected this request under its usage guidelines before generating a "
    "response. xAI charges a $0.05 fee for such requests, so this run is charged."
)


def get_api_key() -> str:
    key = os.environ.get("XAI_API_KEY")
    if not key:
        raise RuntimeError(
            "XAI_API_KEY is not set. A secret whose record exists but holds an empty value is "
            "not injected at all: check that `belt secrets get XAI_API_KEY --json` reports a "
            "non-empty masked_value, and re-set it if it does not."
        )
    return key.strip()


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


def _user_content(text: Optional[str], images, files) -> Any:
    if files:
        raise ValueError(
            "File input is not supported by the Grok apps. Paste the text into the message, "
            "or send images through the images field."
        )
    parts: List[Dict[str, Any]] = []
    if text:
        parts.append({"type": "input_text", "text": text})
    for image in images or []:
        part = _image_part(image)
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
            if getattr(msg, "files", None):
                print(f"Dropping {len(msg.files)} file(s) from an earlier turn: file input is not supported")
            content = _user_content(msg.text, getattr(msg, "images", None), None)
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
    effort = input_data.reasoning_effort
    if isinstance(effort, Enum):
        effort = effort.value
    return effort or None


_EFFORT_ORDER = ["none", "minimal", "low", "medium", "high", "xhigh", "max"]


def build_reasoning(input_data, efforts: Optional[Tuple[str, ...]]) -> Optional[Dict[str, Any]]:
    """Build the `reasoning` request parameter.

    `efforts` is the model's accepted set, lowest first; None means the model
    publishes no effort control and the parameter is never sent. A value
    outside the set maps to the nearest accepted one, so the platform-wide
    default "none" becomes the cheapest valid request (grok-4.7: "low").
    """
    effort = _effort_value(input_data)
    if not effort or not efforts:
        return None
    if effort not in efforts:
        rank = _EFFORT_ORDER.index(effort) if effort in _EFFORT_ORDER else 0
        effort = min(efforts, key=lambda e: abs(_EFFORT_ORDER.index(e) - rank))
    return {"effort": effort}


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
    efforts: Optional[Tuple[str, ...]],
) -> Dict[str, Any]:
    instructions, items = build_input(input_data)
    tools = build_tools(input_data.tools) if input_data.tools else None

    # max_tokens capped at the model limit.
    requested = input_data.max_tokens or max_output_tokens

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

    reasoning = build_reasoning(input_data, efforts)
    if reasoning:
        body["reasoning"] = reasoning

    return body


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

def _format_error(status: int, body: Any, headers: Dict[str, str], prefix: str = "xAI API") -> str:
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


def is_usage_violation(status: int, body: Any) -> bool:
    """True for a 4xx that rejects the request under xAI's usage guidelines."""
    if not 400 <= status < 500 or status == 429:
        return False
    blob = json.dumps(body) if not isinstance(body, str) else body
    blob = blob.lower()
    return any(m in blob for m in _VIOLATION_MARKERS)


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
        "_reasoning_event": None,  # the reasoning delta event type in use
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
        "cost_ticks": None,
        "usage_violation": False,
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

    if etype in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
        text = data.get("delta") or ""
        if not text:
            return None
        # xAI may stream the same reasoning under both event types; keep one.
        if state["_reasoning_event"] is None:
            state["_reasoning_event"] = etype
        elif state["_reasoning_event"] != etype:
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
            # xAI sends the whole call in one chunk, arguments included.
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
        current = state["tool_calls"][idx]["function"]["arguments"]
        if final_args and final_args != current:
            state["tool_calls"][idx]["function"]["arguments"] = final_args
            if final_args.startswith(current):
                return {"tool_calls": [{"index": idx, "function": {"arguments": final_args[len(current):]}}]}
        return None

    if etype in ("response.completed", "response.incomplete"):
        resp = data.get("response") or {}
        state["response_id"] = resp.get("id") or state["response_id"]
        state["status"] = resp.get("status") or etype.split(".")[-1]
        if etype == "response.incomplete":
            state["incomplete_reason"] = (resp.get("incomplete_details") or {}).get("reason") or "incomplete"
        usage = resp.get("usage") or {}
        if usage:
            # Some xAI examples use Chat Completions names; accept both.
            state["input_tokens"] = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
            state["output_tokens"] = usage.get("output_tokens") or usage.get("completion_tokens") or 0
            state["cached_tokens"] = (
                (usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
            )
            state["reasoning_tokens"] = (
                (usage.get("output_tokens_details") or usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
            )
            state["total_tokens"] = usage.get("total_tokens") or 0
            if usage.get("cost_in_usd_ticks") is not None:
                state["cost_ticks"] = usage["cost_in_usd_ticks"]
        return None

    if etype == "response.created":
        resp = data.get("response") or {}
        state["response_id"] = resp.get("id")
        return None

    if etype in ("response.failed", "error"):
        if etype == "response.failed":
            err = (data.get("response") or {}).get("error") or {}
        else:
            err = data.get("error") if isinstance(data.get("error"), dict) else data
        print(f"  xAI {etype} body: {json.dumps(data)[:1000]}")
        if is_usage_violation(400, err) and not state["response"] and not state["tool_calls"]:
            state["usage_violation"] = True
            return None
        raise RuntimeError(f"xAI {etype}: [{err.get('code', '')}] {err.get('message', 'unknown error')}")

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

    output_tokens = billable_output_tokens(state)
    inputs = []
    if state["input_tokens"]:
        inputs.append(TextMeta(tokens=state["input_tokens"], extra={
            "cache_read_tokens": state["cached_tokens"],
        }))
    # Always one output item: it carries the violation flag and xAI's cost even
    # when no tokens were produced. Pricing reads outputs[0].
    outputs = [TextMeta(tokens=output_tokens, extra={
        "reasoning_tokens": state["reasoning_tokens"],
        "usage_violation": 1 if state["usage_violation"] else 0,
        "upstream_cost_usd": state["cost_ticks"] * USD_PER_TICK if state["cost_ticks"] is not None else None,
    })]
    out["output_meta"] = OutputMeta(inputs=inputs, outputs=outputs)
    if state["usage_violation"]:
        out["notice"] = VIOLATION_NOTICE

    elapsed = max(time.monotonic() - state["started_at"], 1e-6)
    ttft = (state["first_token_at"] - state["started_at"]) if state["first_token_at"] else 0.0
    gen_time = max(elapsed - ttft, 1e-6)
    out["usage"] = LLMUsage(
        stop_reason="usage_violation" if state["usage_violation"] else _stop_reason(state),
        time_to_first_token=round(ttft, 3),
        tokens_per_second=round(output_tokens / gen_time, 2) if output_tokens else 0.0,
        prompt_tokens=state["input_tokens"],
        completion_tokens=output_tokens,
        total_tokens=state["input_tokens"] + output_tokens,
        reasoning_tokens=state["reasoning_tokens"],
    )
    return out


def billable_output_tokens(state: Dict[str, Any]) -> int:
    """Output tokens including reasoning.

    xAI's docs disagree on whether usage.output_tokens already includes
    reasoning tokens; total_tokens settles it per response.
    """
    out, reasoning = state["output_tokens"], state["reasoning_tokens"]
    if reasoning and state["total_tokens"] and state["total_tokens"] - state["input_tokens"] == out + reasoning:
        return out + reasoning
    return out


# ---------------------------------------------------------------------------
# Main streaming function
# ---------------------------------------------------------------------------

async def stream_completion(
    input_data,
    model: str,
    *,
    max_output_tokens: int = 128000,
    efforts: Optional[Tuple[str, ...]] = None,
    with_deltas: bool = False,
) -> AsyncGenerator[Any, None]:
    """Stream a Responses API completion.

    with_deltas=False: yields accumulated output dicts (last one carries usage + output_meta).
    with_deltas=True:  yields (output_dict, delta_dict | None) tuples; delta_dict has
                       LLMDelta-shaped keys (response, reasoning, tool_calls).
    """
    body = build_request_body(input_data, model, max_output_tokens=max_output_tokens, efforts=efforts)
    headers = {
        "Authorization": f"Bearer {get_api_key()}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    timeout = httpx.Timeout(connect=15.0, read=None, write=30.0, pool=15.0)
    state = _create_initial_state()
    events = 0

    async with httpx.AsyncClient(timeout=timeout) as http:
        resp = None
        for attempt in range(1, MAX_CONNECT_RETRIES + 1):
            print(
                f"Calling xAI model={model} attempt={attempt} items={len(body['input'])}"
                f" tools={len(body.get('tools') or [])} reasoning={body.get('reasoning')}"
                f" max_output_tokens={body['max_output_tokens']}"
            )
            try:
                req = http.build_request("POST", f"{XAI_BASE_URL}/responses", json=body, headers=headers)
                resp = await http.send(req, stream=True)
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.WriteTimeout, httpx.PoolTimeout) as e:
                if attempt < MAX_CONNECT_RETRIES:
                    print(f"  Connection failed ({type(e).__name__}) on attempt {attempt}, retrying...")
                    await asyncio.sleep(attempt)
                    continue
                raise RuntimeError(f"xAI API connection failed after {attempt} attempts: {e}")

            if resp.status_code == 200:
                break

            raw = await resp.aread()
            await resp.aclose()
            try:
                err_body = json.loads(raw)
            except json.JSONDecodeError:
                err_body = raw.decode("utf-8", "replace")[:500]
            resp_headers = dict(resp.headers)
            if 400 <= resp.status_code < 500:
                # The violation error shape is undocumented: keep the evidence.
                print(f"  HTTP {resp.status_code} body: {json.dumps(err_body)[:1000]}")

            if is_usage_violation(resp.status_code, err_body):
                state["usage_violation"] = True
                resp = None
                break

            if resp.status_code in RETRYABLE_STATUS and attempt < MAX_CONNECT_RETRIES:
                retry_after = resp_headers.get("retry-after")
                backoff = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else 2.0 ** attempt
                print(f"  HTTP {resp.status_code} on attempt {attempt}, backing off {backoff:.0f}s")
                await asyncio.sleep(min(backoff, 30.0))
                continue

            raise RuntimeError(_format_error(resp.status_code, err_body, resp_headers))

        if resp is not None:
            request_id = resp.headers.get("x-request-id", "")
            print(f"Stream opened model={model} req={request_id or 'unknown'}")
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

    if state["usage_violation"]:
        print("xAI rejected the request under its usage guidelines; billing the violation fee")
    if state["incomplete_reason"]:
        print(f"Response incomplete: reason={state['incomplete_reason']}")

    cost = f"${state['cost_ticks'] * USD_PER_TICK:.6f}" if state["cost_ticks"] is not None else "?"
    print(
        f"Stream done model={model} resp={state.get('response_id') or '?'} events={events}"
        f" status={state.get('status')} in={state['input_tokens']} cached={state['cached_tokens']}"
        f" out={state['output_tokens']} reasoning={state['reasoning_tokens']} total={state['total_tokens']}"
        f" billed_out={billable_output_tokens(state)} violation={state['usage_violation']}"
        f" tool_calls={len(state['tool_calls'])} xai_cost={cost}"
    )

    final = _build_output(state, final=True)
    yield (final, None) if with_deltas else final

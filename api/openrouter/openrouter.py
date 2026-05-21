"""OpenRouter stream processing helpers — raw httpx SSE, no OpenAI SDK.

Uses direct HTTP + SSE parsing so we get all OpenRouter-specific fields
(provider, cost, cost_details, reasoning_tokens) that the OpenAI SDK strips.
"""

import asyncio
import json
import logging
import os
import time
from typing import List, Optional, Dict, Any, AsyncGenerator

import httpx

from inferencesh import File, OutputMeta, TextMeta
from inferencesh.models.llm import build_openai_messages, build_tools
from inferencesh.models.output_meta import RawMeta

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Seconds of silence (no chunks AND no keep-alive) before we give up.
# OpenRouter sends `: OPENROUTER PROCESSING` keep-alives, so true silence
# means the connection is dead.
STREAM_SILENCE_TIMEOUT = 30

# Provider health: exclude providers with 5-minute uptime below this.
PROVIDER_MIN_UPTIME_5M = 90.0

# Re-check provider health every 5 minutes.
PROVIDER_HEALTH_TTL = 300

MAX_PROVIDER_RETRIES = 3


# ---------------------------------------------------------------------------
# Provider health
# ---------------------------------------------------------------------------

def _slug_from_tag(tag: str) -> str:
    """Extract provider slug from endpoint tag (e.g. 'deepinfra/fp8' -> 'deepinfra')."""
    return tag.split("/")[0] if "/" in tag else tag


async def _fetch_provider_health(model: str) -> Dict[str, Any]:
    """Query OpenRouter endpoints API and return routing config."""
    try:
        async with httpx.AsyncClient(timeout=10) as http:
            resp = await http.get(f"{OPENROUTER_BASE_URL}/models/{model}/endpoints")
            if resp.status_code != 200:
                print(f"Provider health check for {model}: HTTP {resp.status_code}")
                return {}
            endpoints = resp.json().get("data", {}).get("endpoints", [])
            healthy, unhealthy = [], []
            name_to_slug = {}
            for ep in endpoints:
                name = ep.get("provider_name", "")
                tag = ep.get("tag", "")
                slug = _slug_from_tag(tag)
                uptime = ep.get("uptime_last_5m")
                status = ep.get("status", 0)
                name_to_slug[name] = slug
                up_str = f"{uptime:.1f}%" if uptime is not None else "n/a"
                if status < 0 or (uptime is not None and uptime < PROVIDER_MIN_UPTIME_5M):
                    unhealthy.append(slug)
                    print(f"  SKIP {name} ({slug}): status={status} uptime_5m={up_str}")
                else:
                    healthy.append(slug)
                    print(f"  OK   {name} ({slug}): status={status} uptime_5m={up_str}")
            print(f"Provider health for {model}: {len(healthy)} healthy, {len(unhealthy)} excluded")
            result: Dict[str, Any] = {"_name_to_slug": name_to_slug}
            if unhealthy:
                result["ignore"] = unhealthy
            if healthy:
                result["order"] = healthy
            return result
    except Exception as e:
        print(f"Failed to fetch provider health for {model}: {e}")
        return {}


_health_cache: Dict[str, tuple] = {}


async def get_provider_config(model: str) -> Dict[str, Any]:
    """Return provider routing config with ignore + order, TTL cached."""
    now = time.monotonic()
    cached = _health_cache.get(model)
    if cached and cached[0] > now:
        ttl_left = int(cached[0] - now)
        cfg = cached[1]
        print(f"Provider health for {model}: cached ({ttl_left}s ttl) order={cfg.get('order', [])} ignore={cfg.get('ignore', [])}")
        return cfg
    print(f"Provider health for {model}: checking endpoints...")
    cfg = await _fetch_provider_health(model)
    _health_cache[model] = (now + PROVIDER_HEALTH_TTL, cfg)
    return cfg


# ---------------------------------------------------------------------------
# Reasoning config
# ---------------------------------------------------------------------------

def get_reasoning_config(input_data) -> Optional[Dict[str, Any]]:
    """Build reasoning config for OpenRouter API."""
    reasoning_effort = getattr(input_data, "reasoning_effort", None)
    reasoning_max_tokens = getattr(input_data, "reasoning_max_tokens", None)
    reasoning_exclude = getattr(input_data, "reasoning_exclude", False)

    reasoning_config = {"exclude": reasoning_exclude}

    if reasoning_effort == "none":
        reasoning_config["effort"] = "none"
    elif reasoning_max_tokens is not None and reasoning_max_tokens > 0:
        reasoning_config["max_tokens"] = reasoning_max_tokens
    elif reasoning_effort:
        reasoning_config["effort"] = reasoning_effort
    else:
        return None
    return reasoning_config


# ---------------------------------------------------------------------------
# Model hooks — transform messages/body before sending
# ---------------------------------------------------------------------------
# Each hook is (predicate, transform) where predicate(model) -> bool and
# transform(body, input_data) -> body. Hooks run in order; first match wins.

def _qwen_nothink_hook(body: Dict[str, Any], input_data) -> Dict[str, Any]:
    """Inject /no_think into the last user message for Qwen models when
    reasoning is disabled. OpenRouter's effort param isn't reliably mapped
    for Qwen — the model-level /no_think tag is authoritative."""
    reasoning_effort = getattr(input_data, "reasoning_effort", None)
    if reasoning_effort != "none":
        return body
    messages = body.get("messages", [])
    # Find last user message and append /no_think
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str) and "/no_think" not in content:
                msg["content"] = content + " /no_think"
            elif isinstance(content, list):
                # multimodal: append to last text part
                for part in reversed(content):
                    if isinstance(part, dict) and part.get("type") == "text":
                        if "/no_think" not in part.get("text", ""):
                            part["text"] = part["text"] + " /no_think"
                        break
            break
    return body


# (model_prefix, hook_fn) — checked in order, first match wins
_MODEL_HOOKS = [
    ("qwen/", _qwen_nothink_hook),
]


def _apply_model_hooks(body: Dict[str, Any], input_data, model: str) -> Dict[str, Any]:
    """Apply model-specific hooks to the request body."""
    for prefix, hook in _MODEL_HOOKS:
        if model.startswith(prefix):
            return hook(body, input_data)
    return body


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class ProviderRateLimited(RuntimeError):
    """A specific upstream provider returned 429. Retryable by excluding it."""
    def __init__(self, message: str, provider_slug: str):
        super().__init__(message)
        self.provider_slug = provider_slug


def _provider_name_to_slug(name: str, name_to_slug: Dict[str, str]) -> Optional[str]:
    """Convert a provider display name to its routing slug."""
    if name in name_to_slug:
        return name_to_slug[name]
    return name.lower().replace(" ", "")


def _handle_error_response(
    status_code: int,
    body: Dict[str, Any],
    headers: Dict[str, str],
    name_to_slug: Optional[Dict[str, str]] = None,
    prefix: str = "OpenRouter API",
) -> RuntimeError:
    """Build a descriptive error from an OpenRouter error response."""
    error_obj = body.get("error", {})
    code = error_obj.get("code", status_code)
    msg = error_obj.get("message", f"HTTP {status_code}")
    request_id = headers.get("x-request-id", "")
    gen_id = headers.get("x-generation-id", "")
    provider = headers.get("x-provider", "")

    metadata = error_obj.get("metadata", {})
    provider_name = metadata.get("provider_name", provider or "")

    print(f"{prefix} error: code={code} msg={msg} provider={provider_name} req={request_id} gen={gen_id}")

    # 429 with a known provider → retryable
    if code == 429 and provider_name:
        slug = _provider_name_to_slug(provider_name, name_to_slug or {})
        if slug:
            return ProviderRateLimited(
                f"{prefix} 429 from {provider_name} ({slug}): {msg} [req:{request_id} gen:{gen_id}]",
                provider_slug=slug,
            )

    # Nested provider error
    raw = metadata.get("raw")
    if raw:
        try:
            raw_error = json.loads(raw)
            nested_msg = raw_error.get("error", {}).get("message")
            if nested_msg:
                return RuntimeError(f"{prefix} error ({provider_name}): {nested_msg} [req:{request_id} gen:{gen_id}]")
        except (json.JSONDecodeError, AttributeError):
            pass

    return RuntimeError(f"{prefix} error: {msg} [req:{request_id} gen:{gen_id}]")


# ---------------------------------------------------------------------------
# SSE parsing
# ---------------------------------------------------------------------------

def _parse_sse_chunk(data: Dict[str, Any], state: Dict[str, Any]) -> Optional[str]:
    """Process a parsed SSE data object and update state. Returns finish_reason if present."""

    # Capture generation_id from chunk id
    chunk_id = data.get("id")
    if chunk_id and not state.get("generation_id"):
        state["generation_id"] = chunk_id

    # Capture provider (OpenRouter sends this on every chunk)
    provider = data.get("provider")
    if provider:
        state["provider"] = provider

    # Actual versioned model (e.g. qwen/qwen3-32b-04-28 vs requested qwen/qwen3-32b)
    actual_model = data.get("model")
    if actual_model and not state.get("actual_model"):
        state["actual_model"] = actual_model

    # Check for error in chunk
    error = data.get("error")
    if error:
        if isinstance(error, dict):
            code = error.get("code", "")
            msg = error.get("message", "Unknown error")
            provider_name = error.get("metadata", {}).get("provider_name", "")
            detail = f"[{code}] {msg}" if code else msg
            if provider_name:
                detail += f" (provider: {provider_name})"
        else:
            detail = str(error)
        raise RuntimeError(f"OpenRouter mid-stream error: {detail}")

    # Usage (OpenRouter sends in final chunk, includes cost + cost_details)
    usage = data.get("usage")
    if usage:
        if usage.get("prompt_tokens") is not None:
            state["input_tokens"] = usage["prompt_tokens"]
        if usage.get("completion_tokens") is not None:
            state["output_tokens"] = usage["completion_tokens"]
        if usage.get("cost") is not None:
            state["cost_usd"] = usage["cost"]
        cost_details = usage.get("cost_details")
        if cost_details:
            state["cost_details"] = cost_details
        prompt_details = usage.get("prompt_tokens_details")
        if prompt_details:
            cached = prompt_details.get("cached_tokens")
            if cached:
                state["cached_tokens"] = cached
        completion_details = usage.get("completion_tokens_details")
        if completion_details:
            reasoning_tokens = completion_details.get("reasoning_tokens")
            if reasoning_tokens is not None:
                state["reasoning_tokens"] = reasoning_tokens

    # Choices
    choices = data.get("choices", [])
    if not choices:
        return None

    choice = choices[0]
    finish_reason = choice.get("finish_reason")
    native_finish = choice.get("native_finish_reason")
    if native_finish and native_finish != finish_reason:
        state["native_finish_reason"] = native_finish
    delta = choice.get("delta", {})

    content = delta.get("content")
    if content:
        state["response"] += content

    reasoning = delta.get("reasoning")
    if reasoning:
        state["reasoning"] += reasoning

    reasoning_details = delta.get("reasoning_details")
    if reasoning_details:
        state["reasoning_details"].extend(reasoning_details)

    tool_calls = delta.get("tool_calls")
    if tool_calls:
        for tc in tool_calls:
            _process_tool_call_delta(tc, state["tool_calls"])

    images = delta.get("images")
    if images:
        for img in images:
            url = img.get("image_url", {}).get("url") if isinstance(img, dict) else None
            if url and url not in state["image_urls"]:
                state["image_urls"].append(url)

    if finish_reason == "error":
        raise RuntimeError("OpenRouter stream terminated with finish_reason=error")

    return finish_reason


def _process_tool_call_delta(tc: Dict[str, Any], tool_calls: List[Dict[str, Any]]) -> None:
    """Process a tool call delta dict and update tool_calls list in place."""
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


# ---------------------------------------------------------------------------
# Output building
# ---------------------------------------------------------------------------

def _create_initial_state() -> Dict[str, Any]:
    return {
        "response": "",
        "reasoning": "",
        "reasoning_details": [],
        "tool_calls": [],
        "image_urls": [],
        "input_tokens": 0,
        "output_tokens": 0,
    }


def _build_output(state: Dict[str, Any]) -> Dict[str, Any]:
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

    # output_meta: token usage + upstream cost from stream
    inputs, outputs = [], []
    if state.get("input_tokens"):
        inputs.append(TextMeta(tokens=state["input_tokens"]))
    if state.get("output_tokens"):
        outputs.append(TextMeta(tokens=state["output_tokens"]))

    gen_extra = {}
    if state.get("generation_id"):
        gen_extra["generation_id"] = state["generation_id"]
    if state.get("provider"):
        gen_extra["provider"] = state["provider"]
    if state.get("actual_model"):
        gen_extra["actual_model"] = state["actual_model"]
    if state.get("cost_details"):
        gen_extra["cost_details"] = state["cost_details"]
    if state.get("reasoning_tokens"):
        gen_extra["reasoning_tokens"] = state["reasoning_tokens"]
    if state.get("cached_tokens"):
        gen_extra["cached_tokens"] = state["cached_tokens"]

    cost_usd = state.get("cost_usd")
    if gen_extra or cost_usd is not None:
        cost_cents = (cost_usd * 100) if cost_usd is not None else 0
        inputs.append(RawMeta(cost=cost_cents, extra=gen_extra or None))

    if inputs or outputs:
        out["output_meta"] = OutputMeta(inputs=inputs, outputs=outputs)

    return out


# ---------------------------------------------------------------------------
# Request building
# ---------------------------------------------------------------------------

def _build_request_body(
    input_data,
    model: str,
    provider_routing: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the JSON body for an OpenRouter chat completion request."""
    messages = build_openai_messages(input_data, file_mode="url", image_mode="url")
    tools = build_tools(input_data.tools) if input_data.tools else None

    body: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "stop": ["<end_of_turn>", "<eos>", "<|im_end|>"],
        "max_tokens": 32768,
    }

    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"

    reasoning_config = get_reasoning_config(input_data)
    if reasoning_config:
        body["reasoning"] = reasoning_config

    # Provider routing from live health data
    provider_config: Dict[str, Any] = {"allow_fallbacks": True}
    if provider_routing:
        provider_config.update({k: v for k, v in provider_routing.items() if not k.startswith("_")})
    body["provider"] = provider_config

    # Model-specific hooks (e.g. Qwen /no_think injection)
    body = _apply_model_hooks(body, input_data, model)

    return body


# ---------------------------------------------------------------------------
# Main streaming function
# ---------------------------------------------------------------------------

async def stream_completion(api_key: str, input_data, model: str) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Stream a completion from OpenRouter via raw httpx SSE.

    Uses direct HTTP so we get all OpenRouter-specific fields (provider, cost,
    cost_details) that the OpenAI SDK strips. Always streams — OpenRouter has
    no keep-alive for non-streaming requests.
    """
    routing = await get_provider_config(model)
    name_to_slug = routing.pop("_name_to_slug", {})

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://inference.sh",
        "X-Title": "inference.sh",
    }

    # Retry loop: on 429 from a specific provider, exclude it and retry
    excluded_slugs: List[str] = []
    resp = None

    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10, read=STREAM_SILENCE_TIMEOUT, write=10, pool=10)) as http:
        for attempt in range(1, MAX_PROVIDER_RETRIES + 1):
            retry_routing = dict(routing)
            if excluded_slugs:
                ignore = list(set(retry_routing.get("ignore", []) + excluded_slugs))
                order = [s for s in retry_routing.get("order", []) if s not in excluded_slugs]
                retry_routing["ignore"] = ignore
                if order:
                    retry_routing["order"] = order

            body = _build_request_body(input_data, model, provider_routing=retry_routing)
            print(f"Calling OpenRouter model={model} attempt={attempt} provider={body.get('provider', {})}")

            try:
                req = http.build_request("POST", f"{OPENROUTER_BASE_URL}/chat/completions", json=body, headers=headers)
                resp = await asyncio.wait_for(http.send(req, stream=True), timeout=15.0)
            except asyncio.TimeoutError:
                raise RuntimeError("OpenRouter API call timed out after 15 seconds")

            if resp.status_code != 200:
                error_body = json.loads(await resp.aread())
                resp_headers = dict(resp.headers)
                err = _handle_error_response(resp.status_code, error_body, resp_headers, name_to_slug)
                if isinstance(err, ProviderRateLimited) and attempt < MAX_PROVIDER_RETRIES:
                    print(f"  429 from {err.provider_slug}, retrying without it (attempt {attempt}/{MAX_PROVIDER_RETRIES})")
                    excluded_slugs.append(err.provider_slug)
                    await resp.aclose()
                    continue
                await resp.aclose()
                raise err
            break  # success

        generation_id = resp.headers.get("x-generation-id")
        print(f"Stream opened gen={generation_id or 'unknown'} model={model}")

        state = _create_initial_state()
        if generation_id:
            state["generation_id"] = generation_id
        chunks_received = 0
        last_data_time = time.monotonic()

        try:
            async for line in resp.aiter_lines():
                now = time.monotonic()

                # SSE keep-alive comment — reset timer but don't process
                if line.startswith(":"):
                    last_data_time = now
                    continue

                if not line.startswith("data: "):
                    continue

                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    nfr = state.get("native_finish_reason")
                    nfr_str = f" native_finish={nfr}" if nfr else ""
                    cached = state.get("cached_tokens")
                    cached_str = f" cached_tokens={cached}" if cached else ""
                    print(
                        f"Stream done gen={generation_id} chunks={chunks_received}"
                        f" provider={state.get('provider', '?')}"
                        f" model={state.get('actual_model', '?')}"
                        f" in={state.get('input_tokens', 0)} out={state.get('output_tokens', 0)}"
                        f" reasoning={state.get('reasoning_tokens', 0)}"
                        f" cost_usd={state.get('cost_usd', 'n/a')}"
                        f"{cached_str}{nfr_str}"
                    )
                    break

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    print(f"  Bad SSE JSON: {data_str[:200]}")
                    continue

                last_data_time = now
                chunks_received += 1

                finish_reason = _parse_sse_chunk(data, state)
                yield _build_output(state)

        except httpx.ReadTimeout:
            detail = f"no data for {STREAM_SILENCE_TIMEOUT}s after {chunks_received} chunks"
            if generation_id:
                detail += f" [gen:{generation_id}]"
            raise RuntimeError(f"Stream timed out — {detail}")
        finally:
            await resp.aclose()

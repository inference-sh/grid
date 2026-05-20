"""OpenRouter stream processing helpers for LLM inference."""

import asyncio
import logging
import os
import time
from typing import List, Optional, Dict, Any, AsyncGenerator, Set
from inferencesh import File
from inferencesh.models.llm import build_openai_messages, build_tools
from inferencesh import OutputMeta, TextMeta

logger = logging.getLogger(__name__)

# Seconds of silence (no chunks AND no keep-alive) before we give up on a stream.
# OpenRouter sends `: OPENROUTER PROCESSING` keep-alives, so if we see nothing at
# all for this long the connection is dead — not just slow.
STREAM_SILENCE_TIMEOUT = 30

# Provider health: exclude providers with 5-minute uptime below this threshold.
PROVIDER_MIN_UPTIME_5M = 90.0

# Re-check provider health every 5 minutes.
PROVIDER_HEALTH_TTL = 300

# Cached per-model unhealthy provider sets: {model: (expiry_ts, set_of_names)}
_unhealthy_cache: Dict[str, tuple] = {}


async def _fetch_unhealthy_providers(model: str) -> Set[str]:
    """Query OpenRouter endpoints API and return provider names with low uptime."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=10) as http:
            resp = await http.get(
                f"https://openrouter.ai/api/v1/models/{model}/endpoints",
            )
            if resp.status_code != 200:
                return set()
            endpoints = resp.json().get("data", {}).get("endpoints", [])
            unhealthy = set()
            for ep in endpoints:
                name = ep.get("provider_name", "")
                uptime = ep.get("uptime_last_5m")
                status = ep.get("status", 0)
                # status < 0 means OpenRouter already flagged it as degraded/down
                if status < 0 or (uptime is not None and uptime < PROVIDER_MIN_UPTIME_5M):
                    unhealthy.add(name)
                    logger.info(f"Excluding provider {name} for {model}: status={status} uptime_5m={uptime}")
            return unhealthy
    except Exception as e:
        logger.warning(f"Failed to fetch provider health for {model}: {e}")
        return set()


async def get_unhealthy_providers(model: str) -> List[str]:
    """Return list of provider names to ignore, with a TTL cache."""
    now = time.monotonic()
    cached = _unhealthy_cache.get(model)
    if cached and cached[0] > now:
        return list(cached[1])
    unhealthy = await _fetch_unhealthy_providers(model)
    _unhealthy_cache[model] = (now + PROVIDER_HEALTH_TTL, unhealthy)
    if unhealthy:
        logger.info(f"Unhealthy providers for {model}: {unhealthy}")
    return list(unhealthy)


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


async def lookup_generation(api_key: str, generation_id: str) -> Optional[Dict[str, Any]]:
    """Query OpenRouter generation API to check if a generation completed server-side."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=10) as http:
            resp = await http.get(
                f"https://openrouter.ai/api/v1/generation?id={generation_id}",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            if resp.status_code == 200:
                return resp.json()
    except Exception as e:
        logger.warning(f"Generation lookup failed for {generation_id}: {e}")
    return None


def handle_api_error(e: Exception, prefix: str = "OpenRouter API") -> RuntimeError:
    """Extract error message from API exception, including nested provider errors."""
    if hasattr(e, "response") and e.response is not None:
        try:
            import json
            error_data = e.response.json()
            error_obj = error_data.get("error", {})
            msg = error_obj.get("message", str(e))

            # Grab request/generation IDs from headers for debugging
            headers = e.response.headers if hasattr(e.response, "headers") else {}
            request_id = headers.get("x-request-id", "")
            gen_id = headers.get("x-generation-id", "")
            provider = headers.get("x-provider", "")

            # Extract nested provider error from metadata.raw
            metadata = error_obj.get("metadata", {})
            raw = metadata.get("raw")
            if raw:
                try:
                    raw_error = json.loads(raw)
                    nested_msg = raw_error.get("error", {}).get("message")
                    if nested_msg:
                        provider_name = metadata.get("provider_name", provider or "Provider")
                        return RuntimeError(f"{prefix} error ({provider_name}): {nested_msg} [req:{request_id} gen:{gen_id}]")
                except json.JSONDecodeError:
                    pass

            # Log full error details for debugging opaque 500s
            logger.warning(f"{prefix} error: {msg} | request_id={request_id} generation_id={gen_id} provider={provider} | body={json.dumps(error_data)} | headers={dict(headers)}")

            return RuntimeError(f"{prefix} error: {msg} [req:{request_id} gen:{gen_id}]")
        except Exception:
            pass
    return RuntimeError(f"{prefix} error: {str(e)}")


def check_chunk_error(chunk, prefix: str = "OpenRouter") -> None:
    """Raise if chunk contains an error."""
    if hasattr(chunk, "error") and chunk.error:
        err = chunk.error
        if isinstance(err, dict):
            code = err.get("code", "")
            msg = err.get("message", "Unknown error")
            metadata = err.get("metadata", {})
            provider = metadata.get("provider_name", "")
            raw = metadata.get("raw", "")
            detail = f"{msg}"
            if code:
                detail = f"[{code}] {detail}"
            if provider:
                detail += f" (provider: {provider})"
            if raw:
                logger.warning(f"{prefix} raw upstream error: {raw}")
        else:
            detail = str(err)
        raise RuntimeError(f"{prefix} mid-stream error: {detail}")

    if chunk.choices and len(chunk.choices) > 0:
        if getattr(chunk.choices[0], "finish_reason", None) == "error":
            raise RuntimeError(f"{prefix} stream terminated with finish_reason=error")


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
    # Log debug echo if present
    debug = getattr(chunk, "debug", None)
    if debug:
        import logging, json
        logging.warning(f"OpenRouter debug upstream body: {json.dumps(debug) if isinstance(debug, dict) else debug}")

    check_chunk_error(chunk)

    # Track usage if available - OpenRouter sends in final chunk (may have empty choices)
    usage_attr = getattr(chunk, "usage", None)
    if usage_attr:
        prompt_tokens = getattr(usage_attr, "prompt_tokens", None)
        completion_tokens = getattr(usage_attr, "completion_tokens", None)
        if prompt_tokens is not None:
            state["input_tokens"] = prompt_tokens
        if completion_tokens is not None:
            state["output_tokens"] = completion_tokens

    # Handle usage-only chunk (empty choices)
    if not chunk.choices:
        return None

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
    
    # Add output_meta with token usage
    inputs = []
    outputs = []
    if state.get("input_tokens"):
        inputs.append(TextMeta(tokens=state["input_tokens"]))
    if state.get("output_tokens"):
        outputs.append(TextMeta(tokens=state["output_tokens"]))
    if inputs or outputs:
        out["output_meta"] = OutputMeta(inputs=inputs, outputs=outputs)

    return out


def create_initial_state() -> Dict[str, Any]:
    """Create initial state dict for stream processing."""
    return {
        "response": "",
        "reasoning": "",
        "reasoning_details": [],
        "tool_calls": [],
        "image_urls": [],
        "input_tokens": 0,
        "output_tokens": 0,
    }


def _build_params(input_data, model: str, stream: bool, ignore_providers: Optional[List[str]] = None) -> Dict[str, Any]:
    """Build common request parameters."""
    messages = build_openai_messages(input_data, file_mode="url", image_mode="url")
    tools = build_tools(input_data.tools) if input_data.tools else None

    params = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "extra_headers": {"HTTP-Referer": "https://inference.sh", "X-Title": "inference.sh"},
        "stop": ["<end_of_turn>", "<eos>", "<|im_end|>"],
        "max_tokens": 64000,
    }

    if stream:
        params["stream_options"] = {"include_usage": True}

    if tools:
        params["tools"] = tools
        params["tool_choice"] = "auto"

    extra_body = {}
    reasoning_config = get_reasoning_config(input_data)
    if reasoning_config:
        extra_body["reasoning"] = reasoning_config

    # Provider routing: sort by throughput, auto-exclude unhealthy providers.
    provider_config: Dict[str, Any] = {
        "sort": "throughput",
        "allow_fallbacks": True,
    }
    if ignore_providers:
        provider_config["ignore"] = ignore_providers
    extra_body["provider"] = provider_config

    params["extra_body"] = extra_body

    return params


async def stream_completion(client, input_data, model: str) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Stream a completion from OpenRouter and yield output dicts.

    Always streams, even when the caller wants a single response. OpenRouter sends
    `: OPENROUTER PROCESSING` SSE keep-alive comments during streaming to prevent
    connection drops, but has NO keep-alive for non-streaming requests. If a
    non-streaming HTTP call times out client-side, the generation still completes
    server-side (and you get billed) but the response is lost — there is no way to
    recover it. See: https://openrouter.ai/docs/api/reference/streaming
    """
    ignore = await get_unhealthy_providers(model)
    params = _build_params(input_data, model, stream=True, ignore_providers=ignore)

    # Enable upstream debug echo so we can see what OpenRouter actually sent
    params["extra_body"]["debug"] = {"echo_upstream_body": True}

    try:
        stream = await asyncio.wait_for(client.chat.completions.create(**params), timeout=15.0)
    except asyncio.TimeoutError:
        raise RuntimeError("OpenRouter API call timed out after 15 seconds")
    except Exception as e:
        raise handle_api_error(e)

    # Grab generation ID from response headers for post-hoc debugging
    generation_id = None
    raw_response = getattr(stream, "response", None)
    if raw_response and hasattr(raw_response, "headers"):
        generation_id = raw_response.headers.get("x-generation-id")
        if generation_id:
            logger.info(f"OpenRouter generation_id={generation_id} model={model}")

    state = create_initial_state()
    chunks_received = 0

    async def _iter_with_timeout():
        """Wrap the stream iterator with a per-chunk deadline so we detect
        silence even when no chunks arrive at all (async-for blocks forever)."""
        nonlocal chunks_received
        aiter = stream.__aiter__()
        while True:
            try:
                chunk = await asyncio.wait_for(aiter.__anext__(), timeout=STREAM_SILENCE_TIMEOUT)
                chunks_received += 1
                yield chunk
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                detail = f"no data for {STREAM_SILENCE_TIMEOUT}s after {chunks_received} chunks"
                if generation_id:
                    detail += f" [gen:{generation_id}]"
                    api_key = client.api_key
                    gen_data = await lookup_generation(api_key, generation_id)
                    if gen_data:
                        finish = gen_data.get("finish_reason", "unknown")
                        provider = gen_data.get("provider_name", "unknown")
                        tokens_out = gen_data.get("tokens_completion", 0)
                        logger.warning(
                            f"Stream timeout but generation {generation_id} has "
                            f"finish_reason={finish} provider={provider} "
                            f"tokens_completion={tokens_out}"
                        )
                        detail += f" server_status={finish} provider={provider}"
                raise RuntimeError(f"Stream timed out — {detail}")

    try:
        async for chunk in _iter_with_timeout():
            finish_reason = process_chunk(chunk, state)
            yield build_output(state)
            # Don't break on finish_reason - OpenRouter sends usage in a subsequent chunk
    except RuntimeError:
        raise
    except Exception as e:
        detail = f"after {chunks_received} chunks"
        if generation_id:
            detail += f" [gen:{generation_id}]"
        logger.error(f"Stream error {detail}: {e}")
        raise
    finally:
        if hasattr(stream, "aclose"):
            await stream.aclose()

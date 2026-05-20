"""OpenRouter stream processing helpers for LLM inference."""

import asyncio
import logging
import os
import time
from typing import List, Optional, Dict, Any, AsyncGenerator, Set
from inferencesh import File
from inferencesh.models.llm import build_openai_messages, build_tools
from inferencesh import OutputMeta, TextMeta
from inferencesh.models.output_meta import RawMeta

logger = logging.getLogger(__name__)

# Seconds of silence (no chunks AND no keep-alive) before we give up on a stream.
# OpenRouter sends `: OPENROUTER PROCESSING` keep-alives, so if we see nothing at
# all for this long the connection is dead — not just slow.
STREAM_SILENCE_TIMEOUT = 30

# Provider health: exclude providers with 5-minute uptime below this threshold.
PROVIDER_MIN_UPTIME_5M = 90.0

# Re-check provider health every 5 minutes.
PROVIDER_HEALTH_TTL = 300



def _slug_from_tag(tag: str) -> str:
    """Extract provider slug from endpoint tag (e.g. 'deepinfra/fp8' -> 'deepinfra')."""
    return tag.split("/")[0] if "/" in tag else tag


async def _fetch_provider_health(model: str) -> Dict[str, Any]:
    """Query OpenRouter endpoints API and return routing config.

    Returns {"ignore": [...slugs...], "order": [...slugs...]} based on
    live uptime data. Slugs come from the endpoint tag field, which is
    what OpenRouter's provider routing actually matches on.
    """
    import httpx
    try:
        async with httpx.AsyncClient(timeout=10) as http:
            resp = await http.get(
                f"https://openrouter.ai/api/v1/models/{model}/endpoints",
            )
            if resp.status_code != 200:
                logger.warning(f"Provider health check for {model}: HTTP {resp.status_code}")
                return {}
            endpoints = resp.json().get("data", {}).get("endpoints", [])
            healthy = []
            unhealthy = []
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
        logger.warning(f"Failed to fetch provider health for {model}: {e}")
        return {}


# Cached per-model provider config: {model: (expiry_ts, config_dict)}
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


class ProviderRateLimited(RuntimeError):
    """A specific upstream provider returned 429. Retryable by excluding it."""
    def __init__(self, message: str, provider_slug: str):
        super().__init__(message)
        self.provider_slug = provider_slug


async def lookup_generation(
    api_key: str, generation_id: str, retries: int = 3, delay: float = 2.0,
) -> Optional[Dict[str, Any]]:
    """Query OpenRouter generation API, retrying on 404 (generation not yet indexed)."""
    import httpx
    for attempt in range(retries):
        try:
            if attempt > 0:
                await asyncio.sleep(delay)
            async with httpx.AsyncClient(timeout=10) as http:
                resp = await http.get(
                    f"https://openrouter.ai/api/v1/generation?id={generation_id}",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code != 404:
                    print(f"Generation lookup {generation_id}: HTTP {resp.status_code}")
                    return None
                # 404 — not indexed yet, retry
        except Exception as e:
            print(f"Generation lookup {generation_id} failed: {e}")
            return None
    return None


def _provider_name_to_slug(name: str, name_to_slug: Dict[str, str]) -> Optional[str]:
    """Convert a provider display name to its routing slug."""
    # Try exact match from cached endpoint data first
    if name in name_to_slug:
        return name_to_slug[name]
    # Fallback: lowercase, remove spaces (covers Groq->groq, DeepInfra->deepinfra)
    return name.lower().replace(" ", "")


def handle_api_error(
    e: Exception,
    prefix: str = "OpenRouter API",
    name_to_slug: Optional[Dict[str, str]] = None,
) -> RuntimeError:
    """Extract error message from API exception, including nested provider errors.

    For 429s with a provider_name in metadata, raises ProviderRateLimited so the
    caller can retry with that provider excluded.
    """
    if hasattr(e, "response") and e.response is not None:
        try:
            import json
            error_data = e.response.json()
            error_obj = error_data.get("error", {})
            code = error_obj.get("code", getattr(e.response, "status_code", 0))
            msg = error_obj.get("message", str(e))

            # Grab request/generation IDs from headers for debugging
            headers = e.response.headers if hasattr(e.response, "headers") else {}
            request_id = headers.get("x-request-id", "")
            gen_id = headers.get("x-generation-id", "")
            provider = headers.get("x-provider", "")

            metadata = error_obj.get("metadata", {})
            provider_name = metadata.get("provider_name", provider or "")

            # Log full error details for debugging
            logger.warning(f"{prefix} error: {msg} | request_id={request_id} generation_id={gen_id} provider={provider} | body={json.dumps(error_data)} | headers={dict(headers)}")

            # 429 with a known provider → retryable by excluding that provider
            if code == 429 and provider_name:
                slug = _provider_name_to_slug(provider_name, name_to_slug or {})
                if slug:
                    return ProviderRateLimited(
                        f"{prefix} 429 from {provider_name} ({slug}): {msg} [req:{request_id} gen:{gen_id}]",
                        provider_slug=slug,
                    )

            # Extract nested provider error from metadata.raw
            raw = metadata.get("raw")
            if raw:
                try:
                    raw_error = json.loads(raw)
                    nested_msg = raw_error.get("error", {}).get("message")
                    if nested_msg:
                        return RuntimeError(f"{prefix} error ({provider_name}): {nested_msg} [req:{request_id} gen:{gen_id}]")
                except json.JSONDecodeError:
                    pass

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

    # Capture the chunk id — OpenRouter returns the generation ID here
    chunk_id = getattr(chunk, "id", None)
    if chunk_id and not state.get("_chunk_id"):
        state["_chunk_id"] = chunk_id

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
    
    # Add output_meta with token usage, upstream cost, and provider
    inputs = []
    outputs = []
    if state.get("input_tokens"):
        inputs.append(TextMeta(tokens=state["input_tokens"]))
    if state.get("output_tokens"):
        outputs.append(TextMeta(tokens=state["output_tokens"]))
    # Upstream cost/provider from OpenRouter generation API
    gen_extra = {}
    if state.get("generation_id"):
        gen_extra["generation_id"] = state["generation_id"]
    if state.get("provider"):
        gen_extra["provider"] = state["provider"]
    cost_usd = state.get("cost_usd")
    if gen_extra or cost_usd is not None:
        cost_cents = (cost_usd * 100) if cost_usd is not None else 0
        inputs.append(RawMeta(cost=cost_cents, extra=gen_extra or None))
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


def _build_params(input_data, model: str, stream: bool, provider_routing: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build common request parameters."""
    messages = build_openai_messages(input_data, file_mode="url", image_mode="url")
    tools = build_tools(input_data.tools) if input_data.tools else None

    params = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "extra_headers": {"HTTP-Referer": "https://inference.sh", "X-Title": "inference.sh"},
        "stop": ["<end_of_turn>", "<eos>", "<|im_end|>"],
        "max_tokens": 32768,
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

    # Provider routing: built dynamically from live endpoint health data.
    # order/ignore use provider slugs from the endpoints API tag field.
    provider_config: Dict[str, Any] = {"allow_fallbacks": True}
    if provider_routing:
        # Filter out internal keys (prefixed with _)
        provider_config.update({k: v for k, v in provider_routing.items() if not k.startswith("_")})
    extra_body["provider"] = provider_config

    params["extra_body"] = extra_body

    return params


MAX_PROVIDER_RETRIES = 3


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
    routing = await get_provider_config(model)
    name_to_slug = routing.pop("_name_to_slug", {})

    # Retry loop: on 429 from a specific provider, exclude it and retry
    excluded_slugs: List[str] = []
    for attempt in range(1, MAX_PROVIDER_RETRIES + 1):
        retry_routing = dict(routing)
        if excluded_slugs:
            ignore = list(set(retry_routing.get("ignore", []) + excluded_slugs))
            order = [s for s in retry_routing.get("order", []) if s not in excluded_slugs]
            retry_routing["ignore"] = ignore
            if order:
                retry_routing["order"] = order

        params = _build_params(input_data, model, stream=True, provider_routing=retry_routing)
        params["extra_body"]["debug"] = {"echo_upstream_body": True}

        print(f"Calling OpenRouter model={model} attempt={attempt} provider={params['extra_body'].get('provider', {})}")

        try:
            stream = await asyncio.wait_for(client.chat.completions.create(**params), timeout=15.0)
            break  # success — proceed to streaming
        except asyncio.TimeoutError:
            raise RuntimeError("OpenRouter API call timed out after 15 seconds")
        except Exception as e:
            err = handle_api_error(e, name_to_slug=name_to_slug)
            if isinstance(err, ProviderRateLimited) and attempt < MAX_PROVIDER_RETRIES:
                print(f"  429 from {err.provider_slug}, retrying without it (attempt {attempt}/{MAX_PROVIDER_RETRIES})")
                excluded_slugs.append(err.provider_slug)
                continue
            raise err

    # Grab generation ID from response headers for post-hoc debugging
    generation_id = None
    raw_response = getattr(stream, "response", None)
    if raw_response and hasattr(raw_response, "headers"):
        generation_id = raw_response.headers.get("x-generation-id")
    print(f"Stream opened gen={generation_id or 'unknown'} model={model}")

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
            # Use chunk id as generation_id (OpenRouter puts it there)
            gen_id = generation_id or state.get("_chunk_id")
            if gen_id:
                state["generation_id"] = gen_id

            yield build_output(state)
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

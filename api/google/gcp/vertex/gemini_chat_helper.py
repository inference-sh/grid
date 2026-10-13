"""Google Vertex AI Gemini chat helper for streaming completions.

Shared helper for Gemini chat/text model apps. Handles message conversion,
streaming, tool use, reasoning/thinking, and token usage tracking via the
google-genai SDK with Vertex AI authentication.
"""

import os
import asyncio
import json
import base64
import re
import logging
from typing import AsyncGenerator, List, Optional, Dict, Any, Tuple

from google import genai
from google.genai import types
from google.genai.types import HttpOptions
from google.oauth2.credentials import Credentials

from inferencesh import OutputMeta, TextMeta
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


def create_vertex_client(
    location: Optional[str] = None,
    api_version: str = "v1",
) -> genai.Client:
    access_token = os.environ.get("GCP_ACCESS_TOKEN")
    project = os.environ.get("GCP_PROJECT_NUMBER")
    if not access_token:
        raise RuntimeError("GCP_ACCESS_TOKEN environment variable is required for Vertex AI access.")
    if not project:
        raise RuntimeError("GCP_PROJECT_NUMBER environment variable is required for Vertex AI access.")
    credentials = Credentials(token=access_token)
    client_kwargs = {
        "vertexai": True,
        "project": project,
        "credentials": credentials,
        "http_options": HttpOptions(api_version=api_version),
    }
    if location:
        client_kwargs["location"] = location
    return genai.Client(**client_kwargs)


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(level=level)
    return logging.getLogger(name)

EFFORT_TO_BUDGET = {
    ReasoningEffortEnum.LOW: 1024,
    ReasoningEffortEnum.MEDIUM: 8192,
    ReasoningEffortEnum.HIGH: 32768,
}


def _build_image_part_from_uri(uri: str) -> types.Part:
    if uri.startswith("data:"):
        match = re.match(r"data:([^;]+);base64,(.+)", uri)
        if match:
            mime_type = match.group(1)
            data = base64.b64decode(match.group(2))
            return types.Part.from_bytes(data=data, mime_type=mime_type)
        data = base64.b64decode(uri)
        return types.Part.from_bytes(data=data, mime_type="image/png")
    return types.Part.from_uri(file_uri=uri, mime_type="image/png")


# Gemini scales an image past a 3072px side down, and an inline image is at
# most 7MB (firebase.google.com/docs/ai-logic/input-file-requirements). A
# local image is fitted first so the request carries no bytes the model never
# sees.
IMAGE_LIMITS = ImageLimits(max_long_edge=3072, max_bytes=7 * 1024 * 1024)


def _build_image_part_from_path(path: str) -> types.Part:
    data, mime_type = fit_image(path, IMAGE_LIMITS)
    return types.Part.from_bytes(data=data, mime_type=mime_type)


def _collect_image_parts(images) -> List[types.Part]:
    parts = []
    if not images:
        return parts
    for image in images:
        if image.path:
            parts.append(_build_image_part_from_path(image.path))
        elif image.uri:
            parts.append(_build_image_part_from_uri(image.uri))
    return parts


def _function_response_parts(images) -> Optional[List[types.FunctionResponsePart]]:
    """Images a tool returned, as parts of its FunctionResponse (Gemini 3
    multimodal function responses), where the model reads them as the tool's
    output. Sent without a $ref to them in response: the API refuses that."""
    parts = []
    for image in images or []:
        if image.path:
            data, mime_type = fit_image(image.path, IMAGE_LIMITS)
            parts.append(types.FunctionResponsePart.from_bytes(data=data, mime_type=mime_type))
        elif image.uri and image.uri.startswith("data:"):
            match = re.match(r"data:([^;]+);base64,(.+)", image.uri)
            if match:
                parts.append(types.FunctionResponsePart.from_bytes(
                    data=base64.b64decode(match.group(2)), mime_type=match.group(1)))
        elif image.uri:
            parts.append(types.FunctionResponsePart(file_data=types.FunctionResponseFileData(
                file_uri=image.uri, mime_type=getattr(image, "content_type", None) or "image/png")))
    return parts or None


# The thought signature Vertex documents for a function call whose signature
# is unavailable (docs.cloud.google.com/vertex-ai/generative-ai/docs/thought-signatures):
# Gemini 3 refuses a current-turn function call without one (400). The
# model's own signature is replayed from the message's reasoning state; this
# stands in only where there is none (a message from before the state was
# kept, or from another model). The SDK sends bytes base64url-encoded, so it
# is decoded here to reach the API as the literal.
SKIP_THOUGHT_SIGNATURE = base64.urlsafe_b64decode("skip_thought_signature_validator")

# The key this helper's entry has in an output's reasoning_state. The entry:
# {"model": <model>, "function_calls": {<tool call id>: <signature>},
#  "text": <signature>}, each signature base64 of the bytes Gemini returned.
# Other apps' entries in the same state are never read.
REASONING_STATE_KEY = "gemini"


def _own_signatures(msg, model: str) -> Dict[str, Any]:
    """The signatures this model returned with an assistant message, or {}.

    A signature belongs to the model that made it: one from another Gemini
    model is not replayed."""
    entry = (getattr(msg, "reasoning_state", None) or {}).get(REASONING_STATE_KEY)
    if not isinstance(entry, dict) or entry.get("model") != model:
        return {}
    return entry


def _signature_bytes(value: Optional[str]) -> Optional[bytes]:
    return base64.b64decode(value) if value else None


def _is_gemini_3_or_later(model: str) -> bool:
    match = re.match(r"gemini-(\d+)", model or "")
    return bool(match) and int(match.group(1)) >= 3


def _tool_result_parts(name: str, response: Dict[str, Any], images, native_images: bool) -> Tuple[List[types.Part], List[types.Part]]:
    """A tool result as (function_response parts, image parts for a user turn
    after the tool results).

    Gemini 3 takes the tool's images in the FunctionResponse (multimodal
    function responses). Gemini 2.5 refuses FunctionResponse.parts or ignores
    it, and reads images placed beside a function_response only sometimes; a
    probe on Vertex (2.5 flash, flash-lite and pro, the call replayed as we
    send it) read them every time from a user turn of their own after the
    function responses."""
    if native_images:
        return [types.Part(function_response=types.FunctionResponse(
            name=name, response=response, parts=_function_response_parts(images)))], []
    image_parts = _collect_image_parts(images)
    if image_parts:
        image_parts = [types.Part.from_text(text=f"Images returned by the {name} tool:")] + image_parts
    return [types.Part(function_response=types.FunctionResponse(name=name, response=response))], image_parts


def convert_messages_to_gemini(
    input_data: LLMInput,
    model: str = "",
) -> Tuple[Optional[str], List[types.Content]]:
    """Convert LLMInput to Gemini contents format for model.

    Returns (system_instruction, contents) tuple.
    Gemini uses role="user" and role="model" (not "assistant").
    Tool results are sent as function_response Parts in a "user" turn.
    """
    system_prompt = input_data.system_prompt if input_data.system_prompt else None
    contents: List[types.Content] = []
    gemini_3 = _is_gemini_3_or_later(model)
    # Tool images for the user turn after a run of tool results, so every
    # function response still directly follows the call it answers.
    pending_images: List[types.Part] = []
    # A function response names the function it answers; a tool message
    # carries only the call id, so the name comes from the call.
    call_names: Dict[str, str] = {}
    for m in input_data.context:
        for tc in (m.tool_calls or []) if m.role == ContextMessageRole.ASSISTANT else []:
            if tc.get("id"):
                call_names[tc["id"]] = tc.get("function", {}).get("name", "")

    def flush_images() -> None:
        if pending_images:
            contents.append(types.Content(role="user", parts=list(pending_images)))
            pending_images.clear()

    for msg in input_data.context:
        parts: List[types.Part] = []

        if msg.role == ContextMessageRole.TOOL and msg.tool_call_id:
            try:
                response_data = json.loads(msg.text) if msg.text else {}
            except json.JSONDecodeError:
                response_data = {"result": msg.text or ""}
            tool_name = getattr(msg, "tool_name", None) or call_names.get(msg.tool_call_id) or msg.tool_call_id
            fr_parts, image_parts = _tool_result_parts(tool_name, response_data, getattr(msg, "images", None), gemini_3)
            contents.append(types.Content(role="user", parts=fr_parts))
            pending_images.extend(image_parts)
            continue

        flush_images()

        role = "user" if msg.role == ContextMessageRole.USER else "model"

        own = _own_signatures(msg, model) if msg.role == ContextMessageRole.ASSISTANT else {}

        if msg.text:
            parts.append(types.Part(text=msg.text, thought_signature=_signature_bytes(own.get("text"))))

        parts.extend(_collect_image_parts(getattr(msg, "images", None)))

        if msg.role == ContextMessageRole.ASSISTANT and msg.tool_calls:
            call_signatures = own.get("function_calls") or {}
            for i, tc in enumerate(msg.tool_calls):
                func = tc.get("function", {})
                args_str = func.get("arguments", "{}")
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                except json.JSONDecodeError:
                    args = {}
                # The model signs the first function call of a step; Gemini 3
                # refuses that call unsigned.
                signature = _signature_bytes(call_signatures.get(tc.get("id")))
                if signature is None and gemini_3 and i == 0:
                    signature = SKIP_THOUGHT_SIGNATURE
                parts.append(types.Part(function_call=types.FunctionCall(
                    name=func.get("name", ""),
                    args=args,
                ), thought_signature=signature))

        if parts:
            contents.append(types.Content(role=role, parts=parts))

    # Current user input
    user_parts: List[types.Part] = []
    if hasattr(input_data, "text") and input_data.text:
        user_parts.append(types.Part.from_text(text=input_data.text))
    user_parts.extend(_collect_image_parts(getattr(input_data, "images", None)))

    if hasattr(input_data, "role") and input_data.role == ContextMessageRole.TOOL:
        if hasattr(input_data, "tool_call_id") and input_data.tool_call_id:
            try:
                response_data = json.loads(input_data.text) if input_data.text else {}
            except json.JSONDecodeError:
                response_data = {"result": input_data.text or ""}
            tool_name = getattr(input_data, "tool_name", None) or call_names.get(input_data.tool_call_id) or input_data.tool_call_id
            fr_parts, image_parts = _tool_result_parts(tool_name, response_data, getattr(input_data, "images", None), gemini_3)
            contents.append(types.Content(role="user", parts=fr_parts))
            pending_images.extend(image_parts)
            flush_images()
        elif user_parts:
            flush_images()
            contents.append(types.Content(role="user", parts=user_parts))
    elif user_parts:
        flush_images()
        contents.append(types.Content(role="user", parts=user_parts))
    flush_images()

    return system_prompt, contents


def convert_tools_to_gemini(
    tools: Optional[List[Dict[str, Any]]],
) -> Optional[List[types.Tool]]:
    """Convert OpenAI-format tool definitions to Gemini format."""
    if not tools:
        return None

    declarations = []
    for tool in tools:
        func_def = tool.get("function", tool) if "type" in tool else tool
        params = func_def.get("parameters")

        decl_kwargs = {
            "name": func_def.get("name", ""),
            "description": func_def.get("description", ""),
        }
        if params:
            decl_kwargs["parameters"] = params

        declarations.append(types.FunctionDeclaration(**decl_kwargs))

    return [types.Tool(function_declarations=declarations)]


EFFORT_TO_LEVEL = {
    ReasoningEffortEnum.LOW: types.ThinkingLevel.LOW,
    ReasoningEffortEnum.MEDIUM: types.ThinkingLevel.MEDIUM,
    ReasoningEffortEnum.HIGH: types.ThinkingLevel.HIGH,
}


def build_thinking_config(
    input_data, can_disable_thinking: bool = True, *, thinking_levels: bool = False,
) -> Optional[types.ThinkingConfig]:
    """Map reasoning_effort / reasoning_max_tokens to a ThinkingConfig.

    can_disable_thinking=False for models that reject thinking_budget=0
    (Gemini 2.5 Pro): reasoning_effort=none then leaves the model default.

    thinking_levels=True for models that take thinking_level instead of a
    token budget (Gemini 4). Thinking cannot be turned off there, so
    reasoning_effort=none is the model default and reasoning_max_tokens is
    ignored; low/medium/high map to the level of the same name.
    """
    reasoning_max = getattr(input_data, "reasoning_max_tokens", None)
    reasoning_effort = getattr(input_data, "reasoning_effort", None)

    if thinking_levels:
        level = EFFORT_TO_LEVEL.get(reasoning_effort)
        return types.ThinkingConfig(thinking_level=level) if level else None

    if reasoning_effort == ReasoningEffortEnum.NONE or reasoning_effort == "none":
        return types.ThinkingConfig(thinking_budget=0) if can_disable_thinking else None

    if reasoning_max is not None and reasoning_max > 0:
        return types.ThinkingConfig(thinking_budget=reasoning_max)

    if reasoning_effort and reasoning_effort != ReasoningEffortEnum.NONE:
        budget = EFFORT_TO_BUDGET.get(reasoning_effort, 1024)
        return types.ThinkingConfig(thinking_budget=budget)

    return None


_FUNCTION_CALLING_MODE = {
    ToolChoiceMode.NONE: types.FunctionCallingConfigMode.NONE,
    ToolChoiceMode.AUTO: types.FunctionCallingConfigMode.AUTO,
    ToolChoiceMode.REQUIRED: types.FunctionCallingConfigMode.ANY,
    ToolChoiceMode.FUNCTION: types.FunctionCallingConfigMode.ANY,
}


def build_tool_config(tool_choice: Optional[ToolChoice], has_tools: bool) -> Optional[types.ToolConfig]:
    """LLMInput.tool_choice -> Gemini tool_config.function_calling_config.

    required -> ANY; function -> ANY restricted to allowed_function_names.
    """
    if tool_choice is None:
        return None
    if tool_choice.mode in (ToolChoiceMode.REQUIRED, ToolChoiceMode.FUNCTION) and not has_tools:
        raise ValueError(f"tool_choice={tool_choice.mode.value} requires tools")
    cfg = types.FunctionCallingConfig(mode=_FUNCTION_CALLING_MODE[tool_choice.mode])
    if tool_choice.mode == ToolChoiceMode.FUNCTION:
        if not tool_choice.name:
            raise ValueError("tool_choice.mode=function requires name")
        cfg.allowed_function_names = [tool_choice.name]
    return types.ToolConfig(function_calling_config=cfg)


def build_response_format_config(response_format: Optional[ResponseFormat], has_tools: bool) -> Dict[str, Any]:
    """LLMInput.response_format -> GenerateContentConfig kwargs.

    json_object -> response_mime_type application/json; json_schema adds
    response_json_schema. Gemini does not combine structured output with
    function calling, so that pairing is rejected instead of silently
    dropping one side.
    """
    if response_format is None or response_format.type == ResponseFormatType.TEXT:
        return {}
    if has_tools:
        raise ValueError("response_format json is not supported together with tools on Gemini")
    cfg: Dict[str, Any] = {"response_mime_type": "application/json"}
    if response_format.type == ResponseFormatType.JSON_SCHEMA:
        if not response_format.json_schema:
            raise ValueError("response_format.type=json_schema requires json_schema")
        cfg["response_json_schema"] = response_format.json_schema
    return cfg


def create_initial_state() -> Dict[str, Any]:
    return {
        "response": "",
        "thinking": "",
        "tool_calls": [],
        "input_tokens": 0,
        "output_tokens": 0,
        "thinking_tokens": 0,
        # Thought signatures as returned: tool call id -> signature, and the
        # one on a part that is not a function call.
        "call_signatures": {},
        "text_signature": None,
    }


def apply_parts(parts, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Fold a chunk's parts into state; return the LLMDelta-shaped increment.

    Gemini streams whole function calls, so each becomes one indexed
    tool_call delta carrying the complete arguments JSON.
    """
    delta: Dict[str, Any] = {}
    for part in (parts or []):
        signature = getattr(part, "thought_signature", None)
        if signature and not getattr(part, "function_call", None):
            state["text_signature"] = base64.b64encode(signature).decode()
        if getattr(part, "thought", False) and part.text:
            state["thinking"] += part.text
            delta["reasoning"] = delta.get("reasoning", "") + part.text
        elif part.text is not None:
            state["response"] += part.text
            if part.text:
                delta["response"] = delta.get("response", "") + part.text
        elif getattr(part, "function_call", None):
            fc = part.function_call
            index = len(state["tool_calls"])
            tool_call = {
                "id": f"call_{fc.name}_{index}",
                "type": "function",
                "function": {
                    "name": fc.name,
                    "arguments": json.dumps(dict(fc.args)) if fc.args else "{}",
                },
            }
            state["tool_calls"].append(tool_call)
            if signature:
                state["call_signatures"][tool_call["id"]] = base64.b64encode(signature).decode()
            delta.setdefault("tool_calls", []).append({"index": index, **tool_call})
    return delta or None


def build_output(state: Dict[str, Any], model: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"response": state["response"]}
    if state["thinking"]:
        out["reasoning"] = state["thinking"]
    if state["tool_calls"]:
        out["tool_calls"] = state["tool_calls"]
    if state["call_signatures"] or state["text_signature"]:
        entry: Dict[str, Any] = {"model": model}
        if state["call_signatures"]:
            entry["function_calls"] = dict(state["call_signatures"])
        if state["text_signature"]:
            entry["text"] = state["text_signature"]
        out["reasoning_state"] = {REASONING_STATE_KEY: entry}

    inputs = []
    outputs = []
    if state.get("input_tokens"):
        inputs.append(TextMeta(tokens=state["input_tokens"]))
    if state.get("output_tokens"):
        outputs.append(TextMeta(tokens=state["output_tokens"]))
    if inputs or outputs:
        out["output_meta"] = OutputMeta(inputs=inputs, outputs=outputs)

    return out


async def stream_completion(
    client, input_data, model: str, max_tokens: int = 65536,
    *, can_disable_thinking: bool = True, thinking_levels: bool = False,
    with_deltas: bool = False,
) -> AsyncGenerator[Any, None]:
    """Stream a chat completion from Vertex AI Gemini API.

    with_deltas=False (default): yields accumulated output dicts.
    with_deltas=True: yields (output_dict, delta_dict | None) tuples; the
    delta has LLMDelta-shaped keys (response, reasoning, tool_calls).
    """
    system_prompt, contents = convert_messages_to_gemini(input_data, model)
    tools = convert_tools_to_gemini(input_data.tools) if input_data.tools else None

    config_kwargs: Dict[str, Any] = {
        "max_output_tokens": max_tokens,
    }

    temperature = getattr(input_data, "temperature", None)
    if temperature is not None:
        config_kwargs["temperature"] = temperature
    top_p = getattr(input_data, "top_p", None)
    if top_p is not None:
        config_kwargs["top_p"] = top_p
    top_k = getattr(input_data, "top_k", None)
    if top_k is not None and top_k >= 0:
        config_kwargs["top_k"] = top_k

    if system_prompt:
        config_kwargs["system_instruction"] = system_prompt

    if tools:
        config_kwargs["tools"] = tools
    tool_config = build_tool_config(input_data.tool_choice, has_tools=bool(tools))
    if tool_config is not None:
        config_kwargs["tool_config"] = tool_config
    config_kwargs.update(build_response_format_config(input_data.response_format, has_tools=bool(tools)))

    thinking_config = build_thinking_config(
        input_data, can_disable_thinking=can_disable_thinking, thinking_levels=thinking_levels,
    )
    if thinking_config:
        config_kwargs["thinking_config"] = thinking_config

    config = types.GenerateContentConfig(**config_kwargs)

    logger.info(f"Calling Vertex AI Gemini: model={model}, messages={len(contents)}")

    state = create_initial_state()
    last_chunk_time = asyncio.get_event_loop().time()

    try:
        stream = await client.aio.models.generate_content_stream(
            model=model,
            contents=contents,
            config=config,
        )

        async for chunk in stream:
            now = asyncio.get_event_loop().time()
            if now - last_chunk_time > 120.0:
                raise RuntimeError("Stream timed out - no events for 120 seconds")
            last_chunk_time = now

            # Update usage metadata from any chunk that has it
            usage = getattr(chunk, "usage_metadata", None)
            if usage:
                state["input_tokens"] = getattr(usage, "prompt_token_count", 0) or 0
                candidates_tokens = getattr(usage, "candidates_token_count", 0) or 0
                thinking_tokens = getattr(usage, "thoughts_token_count", 0) or 0
                state["output_tokens"] = candidates_tokens + thinking_tokens
                state["thinking_tokens"] = thinking_tokens

            if not chunk.candidates:
                continue

            candidate = chunk.candidates[0]
            parts = getattr(candidate.content, "parts", None) if candidate.content else None

            delta = apply_parts(parts, state)
            output = build_output(state, model)
            yield (output, delta) if with_deltas else output

    except Exception as e:
        error_str = str(e)
        if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
            raise RuntimeError("Vertex AI rate limit exceeded, please try again later")
        raise

    logger.info(
        f"Vertex AI stream complete: in={state['input_tokens']} out={state['output_tokens']} "
        f"thinking={state['thinking_tokens']}"
    )

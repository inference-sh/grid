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


def _build_image_part_from_path(path: str) -> types.Part:
    with open(path, "rb") as f:
        image_data = f.read()
    ext = path.lower().rsplit(".", 1)[-1]
    mime_type = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "webp": "image/webp",
    }.get(ext, "image/png")
    return types.Part.from_bytes(data=image_data, mime_type=mime_type)


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


def convert_messages_to_gemini(
    input_data: LLMInput,
) -> Tuple[Optional[str], List[types.Content]]:
    """Convert LLMInput to Gemini contents format.

    Returns (system_instruction, contents) tuple.
    Gemini uses role="user" and role="model" (not "assistant").
    Tool results are sent as function_response Parts in a "user" turn.
    """
    system_prompt = input_data.system_prompt if input_data.system_prompt else None
    contents: List[types.Content] = []

    for msg in input_data.context:
        parts: List[types.Part] = []

        if msg.role == ContextMessageRole.TOOL and msg.tool_call_id:
            try:
                response_data = json.loads(msg.text) if msg.text else {}
            except json.JSONDecodeError:
                response_data = {"result": msg.text or ""}
            tool_name = getattr(msg, "tool_name", None) or msg.tool_call_id
            parts.append(types.Part(function_response=types.FunctionResponse(
                name=tool_name,
                response=response_data,
            )))
            contents.append(types.Content(role="user", parts=parts))
            continue

        role = "user" if msg.role == ContextMessageRole.USER else "model"

        if msg.text:
            parts.append(types.Part.from_text(text=msg.text))

        parts.extend(_collect_image_parts(getattr(msg, "images", None)))

        if msg.role == ContextMessageRole.ASSISTANT and msg.tool_calls:
            for tc in msg.tool_calls:
                func = tc.get("function", {})
                args_str = func.get("arguments", "{}")
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                except json.JSONDecodeError:
                    args = {}
                parts.append(types.Part(function_call=types.FunctionCall(
                    name=func.get("name", ""),
                    args=args,
                )))

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
            tool_name = getattr(input_data, "tool_name", None) or input_data.tool_call_id
            tool_parts = [types.Part(function_response=types.FunctionResponse(
                name=tool_name,
                response=response_data,
            ))]
            contents.append(types.Content(role="user", parts=tool_parts))
        elif user_parts:
            contents.append(types.Content(role="user", parts=user_parts))
    elif user_parts:
        contents.append(types.Content(role="user", parts=user_parts))

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


def build_thinking_config(input_data) -> Optional[types.ThinkingConfig]:
    reasoning_max = getattr(input_data, "reasoning_max_tokens", None)
    reasoning_effort = getattr(input_data, "reasoning_effort", None)

    if reasoning_effort == ReasoningEffortEnum.NONE or reasoning_effort == "none":
        return None

    if reasoning_max is not None and reasoning_max > 0:
        return types.ThinkingConfig(thinking_budget=reasoning_max)

    if reasoning_effort and reasoning_effort != ReasoningEffortEnum.NONE:
        budget = EFFORT_TO_BUDGET.get(reasoning_effort, 1024)
        return types.ThinkingConfig(thinking_budget=budget)

    return None


def create_initial_state() -> Dict[str, Any]:
    return {
        "response": "",
        "thinking": "",
        "tool_calls": [],
        "input_tokens": 0,
        "output_tokens": 0,
        "thinking_tokens": 0,
    }


def build_output(state: Dict[str, Any]) -> Dict[str, Any]:
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


async def stream_completion(
    client, input_data, model: str, max_tokens: int = 65536
) -> AsyncGenerator[Dict[str, Any], None]:
    """Stream a chat completion from Vertex AI Gemini API."""
    system_prompt, contents = convert_messages_to_gemini(input_data)
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

    thinking_config = build_thinking_config(input_data)
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

            for part in (parts or []):
                if getattr(part, "thought", False) and part.text:
                    state["thinking"] += part.text
                elif part.text is not None:
                    state["response"] += part.text
                elif getattr(part, "function_call", None):
                    fc = part.function_call
                    tool_call = {
                        "id": f"call_{fc.name}_{len(state['tool_calls'])}",
                        "type": "function",
                        "function": {
                            "name": fc.name,
                            "arguments": json.dumps(dict(fc.args)) if fc.args else "{}",
                        },
                    }
                    state["tool_calls"].append(tool_call)

            yield build_output(state)

    except Exception as e:
        error_str = str(e)
        if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
            raise RuntimeError("Vertex AI rate limit exceeded, please try again later")
        raise

    logger.info(
        f"Vertex AI stream complete: in={state['input_tokens']} out={state['output_tokens']} "
        f"thinking={state['thinking_tokens']}"
    )

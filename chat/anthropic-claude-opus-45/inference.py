import os
import asyncio
import json
import base64
import re
from typing import AsyncGenerator, List, Optional, Dict, Any
from pydantic import Field

from inferencesh import BaseApp
from inferencesh.models.llm import (
    LLMInput,
    LLMOutput,
    ReasoningCapabilityMixin,
    ReasoningMixin,
    ReasoningEffortEnum,
    ToolsCapabilityMixin,
    ToolCallsMixin,
    ContextMessageRole,
)
from anthropic import AsyncAnthropic

# Anthropic configuration
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
DEFAULT_MODEL = "claude-opus-4-5-20251101"

# Budget tokens mapping for effort levels
EFFORT_TO_BUDGET = {
    ReasoningEffortEnum.LOW: 1024,
    ReasoningEffortEnum.MEDIUM: 10240,
    ReasoningEffortEnum.HIGH: 32000,
}


class AppInput(LLMInput, ReasoningCapabilityMixin, ToolsCapabilityMixin):
    """Anthropic input model with extended thinking and tools support."""

    context_size: int = Field(
        default=200000, description="The context size for the model."
    )


class AppOutput(ReasoningMixin, ToolCallsMixin, LLMOutput):
    """Anthropic output model with reasoning, tool calls, and usage information."""

    pass


# --- Message/Tool Conversion Helpers ---


def extract_base64_from_data_uri(data_uri: str) -> tuple[str, str]:
    """Extract media type and base64 data from a data URI."""
    match = re.match(r"data:([^;]+);base64,(.+)", data_uri)
    if match:
        return match.group(1), match.group(2)
    return "image/png", data_uri  # fallback


def convert_messages_to_anthropic(
    input_data: LLMInput,
) -> tuple[Optional[str], List[Dict[str, Any]]]:
    """Convert LLMInput to Anthropic message format.

    Returns (system_prompt, messages) tuple.
    """
    system_prompt = input_data.system_prompt if input_data.system_prompt else None
    messages = []

    # Process context messages
    for msg in input_data.context:
        role = "user" if msg.role == ContextMessageRole.USER else "assistant"

        # Handle tool result messages
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

        # Build content blocks
        content = []

        # Add text content
        if msg.text:
            content.append({"type": "text", "text": msg.text})

        # Add images in Anthropic format
        if msg.images:
            for image in msg.images:
                if image.path:
                    # Read and encode image from path
                    with open(image.path, "rb") as f:
                        image_data = base64.b64encode(f.read()).decode("utf-8")
                    # Detect media type from extension
                    ext = image.path.lower().split(".")[-1]
                    media_type = {
                        "png": "image/png",
                        "jpg": "image/jpeg",
                        "jpeg": "image/jpeg",
                        "gif": "image/gif",
                        "webp": "image/webp",
                    }.get(ext, "image/png")
                    content.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            },
                        }
                    )
                elif image.uri:
                    if image.uri.startswith("data:"):
                        media_type, image_data = extract_base64_from_data_uri(image.uri)
                        content.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_data,
                                },
                            }
                        )
                    else:
                        # URL-based image
                        content.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "url",
                                    "url": image.uri,
                                },
                            }
                        )

        # Handle assistant messages with tool calls
        if msg.role == ContextMessageRole.ASSISTANT and msg.tool_calls:
            for tool_call in msg.tool_calls:
                # Parse function arguments
                func = tool_call.get("function", {})
                args_str = func.get("arguments", "{}")
                try:
                    args = (
                        json.loads(args_str) if isinstance(args_str, str) else args_str
                    )
                except json.JSONDecodeError:
                    args = {}

                content.append(
                    {
                        "type": "tool_use",
                        "id": tool_call.get("id", ""),
                        "name": func.get("name", ""),
                        "input": args,
                    }
                )

        if content:
            messages.append({"role": role, "content": content})

    # Add current user input
    user_content = []
    if hasattr(input_data, "text") and input_data.text:
        user_content.append({"type": "text", "text": input_data.text})

    if hasattr(input_data, "images") and input_data.images:
        for image in input_data.images:
            if image.path:
                with open(image.path, "rb") as f:
                    image_data = base64.b64encode(f.read()).decode("utf-8")
                ext = image.path.lower().split(".")[-1]
                media_type = {
                    "png": "image/png",
                    "jpg": "image/jpeg",
                    "jpeg": "image/jpeg",
                    "gif": "image/gif",
                    "webp": "image/webp",
                }.get(ext, "image/png")
                user_content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        },
                    }
                )
            elif image.uri:
                if image.uri.startswith("data:"):
                    media_type, image_data = extract_base64_from_data_uri(image.uri)
                    user_content.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            },
                        }
                    )
                else:
                    user_content.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": image.uri,
                            },
                        }
                    )

    # Handle tool result input
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
        else:
            if user_content:
                messages.append({"role": "user", "content": user_content})
    elif user_content:
        messages.append({"role": "user", "content": user_content})

    # Merge consecutive messages with same role (Anthropic requirement)
    merged = []
    for msg in messages:
        if merged and merged[-1]["role"] == msg["role"]:
            # Merge content
            if isinstance(merged[-1]["content"], list) and isinstance(
                msg["content"], list
            ):
                merged[-1]["content"].extend(msg["content"])
            elif isinstance(merged[-1]["content"], str) and isinstance(
                msg["content"], str
            ):
                merged[-1]["content"] += "\n" + msg["content"]
            else:
                # Convert to list format and merge
                prev_content = (
                    merged[-1]["content"]
                    if isinstance(merged[-1]["content"], list)
                    else [{"type": "text", "text": merged[-1]["content"]}]
                )
                new_content = (
                    msg["content"]
                    if isinstance(msg["content"], list)
                    else [{"type": "text", "text": msg["content"]}]
                )
                merged[-1]["content"] = prev_content + new_content
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
        # Extract function definition
        if "type" in tool and "function" in tool:
            func_def = tool["function"]
        else:
            func_def = tool

        anthropic_tool = {
            "name": func_def.get("name", ""),
            "description": func_def.get("description", ""),
            "input_schema": func_def.get(
                "parameters", {"type": "object", "properties": {}}
            ),
        }
        result.append(anthropic_tool)

    return result


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


# --- Main App ---


class App(BaseApp):
    def __init__(self):
        super().__init__()
        self.client = None

    async def setup(self, metadata):
        """Initialize the Anthropic client."""
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")

        self.client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        print("Anthropic client initialization complete!")

    async def run(
        self, input_data: AppInput, metadata
    ) -> AsyncGenerator[AppOutput, None]:
        """Run inference using Anthropic API."""
        if not self.client:
            raise RuntimeError("Anthropic client not initialized. Call setup() first.")

        # Convert messages and tools to Anthropic format
        system_prompt, messages = convert_messages_to_anthropic(input_data)
        tools = (
            convert_tools_to_anthropic(input_data.tools) if input_data.tools else None
        )

        # Prepare completion parameters
        params = {
            "model": DEFAULT_MODEL,
            "messages": messages,
            "max_tokens": 64000,
            "stream": True,
        }

        if system_prompt:
            params["system"] = system_prompt

        if tools:
            params["tools"] = tools

        # Add extended thinking configuration if enabled
        # Map reasoning_effort/reasoning_max_tokens to Anthropic's budget_tokens
        if (
            input_data.reasoning_max_tokens is None
            or input_data.reasoning_max_tokens == 0
        ):
            params["thinking"] = {
                "type": "disabled",
            }
        elif input_data.reasoning_effort != ReasoningEffortEnum.NONE:
            if (
                input_data.reasoning_max_tokens is not None
                and input_data.reasoning_max_tokens > 1024
            ):
                budget = input_data.reasoning_max_tokens
            else:
                budget = EFFORT_TO_BUDGET.get(input_data.reasoning_effort, 1024)

            params["thinking"] = {
                "type": "enabled",
                "budget_tokens": budget,
            }
        else:
            params["thinking"] = {
                "type": "disabled",
            }

        # Create stream with timeout
        try:
            stream = await asyncio.wait_for(
                self.client.messages.create(**params), timeout=30.0
            )
        except asyncio.TimeoutError:
            raise RuntimeError("Anthropic API call timed out after 30 seconds")
        except Exception as e:
            raise handle_api_error(e)

        # Process stream
        state = {
            "response": "",
            "thinking": "",
            "tool_calls": [],
            "current_tool": None,
        }
        last_chunk_time = asyncio.get_event_loop().time()

        try:
            async for event in stream:
                now = asyncio.get_event_loop().time()
                if now - last_chunk_time > 120.0:
                    raise RuntimeError(
                        "Stream timed out - no events received for 120 seconds"
                    )
                last_chunk_time = now

                # Handle different event types
                if event.type == "content_block_start":
                    block = event.content_block
                    if block.type == "tool_use":
                        state["current_tool"] = {
                            "id": block.id,
                            "type": "function",
                            "function": {
                                "name": block.name,
                                "arguments": "",
                            },
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
                            state["current_tool"]["function"]["arguments"] += (
                                delta.partial_json
                            )

                elif event.type == "content_block_stop":
                    state["current_tool"] = None

                elif event.type == "message_stop":
                    pass  # Final event

                # Build and yield output
                output_data = {"response": state["response"]}
                if state["thinking"]:
                    output_data["reasoning"] = state["thinking"]
                if state["tool_calls"]:
                    output_data["tool_calls"] = state["tool_calls"]

                yield AppOutput(**output_data)

        except Exception as e:
            if "overloaded" in str(e).lower():
                raise RuntimeError(
                    "Anthropic API is overloaded, please try again later"
                )
            raise

    async def unload(self):
        """Clean up resources."""
        if self.client:
            await self.client.close()
        self.client = None

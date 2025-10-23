import os
import asyncio
from typing import AsyncGenerator
from pydantic import Field

from inferencesh import BaseApp
from inferencesh.models.llm import (
    LLMInput,
    LLMOutput,
    ReasoningCapabilityMixin,
    ReasoningMixin,
    ToolsCapabilityMixin,
    ToolCallsMixin,
    ImageCapabilityMixin,
    build_messages,
    build_tools,
)
from openai import AsyncOpenAI
import base64

# OpenRouter configuration
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Default model configuration
DEFAULT_MODEL = "anthropic/claude-3.5-sonnet"

models = [
    "x-ai/grok-code-fast-1",
    "x-ai/grok-4-fast",
    "anthropic/claude-sonnet-4.5",
    "anthropic/claude-sonnet-4",
    "anthropic/claude-3.7-sonnet",
    "anthropic/claude-3.5-sonnet",
    "z-ai/glm-4.6",
    "z-ai/glm-4.5-air:free",
    "z-ai/glm-4.5",
    "z-ai/glm-4.5-air",
    "google/gemini-2.5-pro",
    "google/gemini-2.5-flash",
    "google/gemini-2.0-flash-001",
    "openai/gpt-5",
    "openai/gpt-5-mini",
    "openai/gpt-5-nano",
    "openai/gpt-oss-20b",
    "openai/gpt-4.1-mini",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "qwen/qwen3-vl-30b-a3b-thinking",
    "qwen/qwen3-vl-235b-a22b-instruct",
    "tngtech/deepseek-r1t2-chimera"
]


class AppInput(LLMInput, ReasoningCapabilityMixin, ToolsCapabilityMixin, ImageCapabilityMixin):
    """OpenRouter input model with reasoning and tools support."""

    model: str = Field(
        default=DEFAULT_MODEL,
        description="The model to use for the OpenRouter API.",
        title="OpenRouterModelEnum",
        enum=models,
    )
    reasoning_effort: str = Field(
        default="high",
        description="Reasoning effort level: 'high', 'medium', or 'low'",
        enum=["high", "medium", "low"],
    )
    reasoning_max_tokens: int = Field(
        default=None,
        description="Maximum tokens to allocate for reasoning (overrides effort)",
    )
    reasoning_exclude: bool = Field(
        default=False, description="Exclude reasoning tokens from response"
    )


class AppOutput(ReasoningMixin, ToolCallsMixin, LLMOutput):
    """OpenRouter output model with reasoning, tool calls, and usage information."""

    pass


def image_to_base64_data_uri(file_path: str) -> str:
    """Convert image file to base64 data URI."""
    with open(file_path, "rb") as img_file:
        base64_data = base64.b64encode(img_file.read()).decode("utf-8")
        return f"data:image/png;base64,{base64_data}"


class App(BaseApp):
    def __init__(self):
        super().__init__()
        self.client = None
        self.model_name = None

    async def setup(self, metadata):
        """Initialize the OpenRouter client and model configuration."""
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")

        # Initialize AsyncOpenAI client with OpenRouter endpoint and simple timeout
        self.client = AsyncOpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
            timeout=15.0,
        )

        print("OpenRouter client initialization complete!")

    async def run(
        self, input_data: AppInput, metadata
    ) -> AsyncGenerator[AppOutput, None]:
        """Run inference using OpenRouter API."""
        if not self.client:
            raise RuntimeError("OpenRouter client not initialized. Call setup() first.")

        try:
            print(
                f"[DEBUG] Input: model={input_data.model}, tools={len(input_data.tools) if input_data.tools else 0}, reasoning={input_data.reasoning}"
            )

            # Build messages and tools using SDK helpers - already in OpenAI format!
            messages = build_messages(input_data)
            tools = build_tools(input_data.tools)

            # Log prepared request
            print("[DEBUG] Prepared request:")
            print(f"[DEBUG]   Messages: {len(messages)}")
            for i, msg in enumerate(messages):
                role = msg.get("role")
                content_preview = str(msg.get("content", ""))[:100]
                has_tool_calls = "tool_calls" in msg and msg["tool_calls"]
                tool_call_id = msg.get("tool_call_id")
                print(
                    f"[DEBUG]     [{i}] role={role}, content={content_preview}..., tool_calls={has_tool_calls}, tool_call_id={tool_call_id}"
                )
            if tools:
                print(f"[DEBUG]   Tools: {len(tools)}")
                for i, tool in enumerate(tools):
                    name = tool.get("function", {}).get("name", "unknown")
                    print(f"[DEBUG]     [{i}] {name}")

            # Prepare completion parameters
            completion_params = {
                "model": input_data.model,
                "messages": messages,
                "temperature": input_data.temperature,
                "top_p": input_data.top_p,
                "stream": True,
                "extra_headers": {
                    "HTTP-Referer": "https://inference.sh",
                    "X-Title": "Inference.sh OpenRouter App",
                },
            }

            # Add tools if provided
            if tools:
                completion_params["tools"] = tools
                completion_params["tool_choice"] = "auto"

            # Add reasoning configuration if enabled
            if input_data.reasoning:
                reasoning_config = {"exclude": input_data.reasoning_exclude}

                # Use max_tokens if specified, otherwise use effort
                if input_data.reasoning_max_tokens is not None:
                    reasoning_config["max_tokens"] = input_data.reasoning_max_tokens
                else:
                    reasoning_config["effort"] = input_data.reasoning_effort

                completion_params["reasoning"] = reasoning_config

            # Add stop sequences
            completion_params["stop"] = ["<end_of_turn>", "<eos>", "<|im_end|>"]

            # Stream the completion with proper error handling and timeout
            print("[DEBUG] About to call OpenRouter API...")
            
            print(f"[DEBUG] Completion params: {completion_params}")
            try:
                # Simple per-request timeout via client
                print("[DEBUG] Starting stream creation with 15s timeout...")
                stream = await self.client.chat.completions.create(
                    timeout=15.0,
                    **completion_params,
                )
                print("[DEBUG] Stream created successfully!")
            except Exception as e:
                # Handle pre-stream errors (HTTP status errors)
                print(f"[DEBUG] API call failed with error: {type(e).__name__}: {str(e)}")
                if hasattr(e, "response") and e.response is not None:
                    try:
                        error_data = e.response.json()
                        error_message = error_data.get("error", {}).get(
                            "message", str(e)
                        )
                        raise RuntimeError(f"OpenRouter API error: {error_message}")
                    except Exception:
                        raise RuntimeError(f"OpenRouter API error: {str(e)}")
                else:
                    raise RuntimeError(f"OpenRouter API error: {str(e)}")

            response_content = ""
            tool_calls = []
            reasoning_content = ""
            reasoning_details = []
            chunk_count = 0
            last_chunk_time = asyncio.get_event_loop().time()

            print("[DEBUG] Starting to process stream chunks...")
            try:
                async for chunk in stream:
                    chunk_count += 1
                    current_time = asyncio.get_event_loop().time()
                    
                    # Check for timeout between chunks (15 seconds)
                    if current_time - last_chunk_time > 120.0:
                        raise RuntimeError("Stream timed out - no chunks received for 120 seconds")
                    
                    last_chunk_time = current_time
                    print(f"[DEBUG] Processing chunk {chunk_count} at {current_time:.2f}")
                    
                    # Handle mid-stream errors (as per OpenRouter documentation)
                    if hasattr(chunk, "error") and chunk.error:
                        error_message = (
                            chunk.error.get("message", "Unknown error")
                            if isinstance(chunk.error, dict)
                            else str(chunk.error)
                        )
                        raise RuntimeError(f"OpenRouter mid-stream error: {error_message}")

                    # Check for error in choices (alternative error format)
                    if chunk.choices and len(chunk.choices) > 0:
                        choice = chunk.choices[0]
                        if (
                            hasattr(choice, "finish_reason")
                            and choice.finish_reason == "error"
                        ):
                            raise RuntimeError("OpenRouter stream terminated due to error")

                    delta = chunk.choices[0].delta
                    finish_reason = chunk.choices[0].finish_reason

                    # Handle content
                    if delta.content:
                        response_content += delta.content

                    # Handle OpenRouter reasoning tokens (new format)
                    if hasattr(delta, "reasoning") and delta.reasoning:
                        reasoning_content += delta.reasoning

                    # Handle reasoning_details (structured reasoning blocks)
                    if hasattr(delta, "reasoning_details") and delta.reasoning_details:
                        reasoning_details.extend(delta.reasoning_details)

                    # Handle tool calls
                    if delta.tool_calls:
                        for tool_call_delta in delta.tool_calls:
                            # Find or create tool call
                            tool_id = tool_call_delta.id
                            if tool_id:
                                # New tool call with ID
                                current_tool = next(
                                    (t for t in tool_calls if t["id"] == tool_id), None
                                )
                                if not current_tool:
                                    current_tool = {
                                        "id": tool_id,
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""},
                                    }
                                    tool_calls.append(current_tool)
                            else:
                                # No ID means this is a continuation (argument chunk) of the last tool call
                                current_tool = tool_calls[-1] if tool_calls else None

                            # Update tool call
                            if current_tool and tool_call_delta.function:
                                if tool_call_delta.function.name:
                                    current_tool["function"]["name"] = (
                                        tool_call_delta.function.name
                                    )
                                if tool_call_delta.function.arguments:
                                    current_tool["function"]["arguments"] += (
                                        tool_call_delta.function.arguments
                                    )

                    # Create output for this chunk - yield for every chunk to provide real-time updates
                    output_data = {
                        "response": response_content,
                    }

                    # Add reasoning if present (OpenRouter format)
                    if reasoning_content:
                        output_data["reasoning"] = reasoning_content

                    # Add reasoning_details if present (structured reasoning blocks)
                    if reasoning_details:
                        output_data["reasoning_details"] = reasoning_details

                    # Add tool calls if present
                    if tool_calls:
                        output_data["tool_calls"] = tool_calls

                    # Yield the output for this chunk
                    yield AppOutput(**output_data)

                    # Check if finished
                    if finish_reason:
                        break
            finally:
                # Ensure stream is properly closed even on error
                if hasattr(stream, 'aclose'):
                    await stream.aclose()

            # Log final response summary
            print(
                f"[DEBUG] Response complete: chunks={chunk_count}, content_len={len(response_content)}, tool_calls={len(tool_calls)}"
            )
            if tool_calls:
                for i, tc in enumerate(tool_calls):
                    name = tc.get("function", {}).get("name", "unknown")
                    args_len = len(tc.get("function", {}).get("arguments", ""))
                    print(f"[DEBUG]   Tool call [{i}]: {name}, args_len={args_len}")

        except Exception as e:
            print(
                f"[ERROR] Exception caught in run method: {type(e).__name__}: {str(e)}"
            )
            raise

    async def unload(self):
        """Clean up resources."""
        if self.client:
            # OpenAI client doesn't need explicit cleanup
            self.client = None

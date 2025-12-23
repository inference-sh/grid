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
    build_messages,
    build_tools,
)
from openai import AsyncOpenAI
import base64

# OpenRouter configuration
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Default model configuration
DEFAULT_MODEL = "z-ai/glm-4.6"

class AppInput(LLMInput, ReasoningCapabilityMixin, ToolsCapabilityMixin):
    """OpenRouter input model with reasoning and tools support."""
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

        # Initialize OpenAI client with OpenRouter endpoint
        self.client = AsyncOpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
        )

        print("OpenRouter client initialization complete!")

    async def run(
        self, input_data: AppInput, metadata
    ) -> AsyncGenerator[AppOutput, None]:
        """Run inference using OpenRouter API."""
        if not self.client:
            raise RuntimeError("OpenRouter client not initialized. Call setup() first.")

        try:
            # Build messages and tools using SDK helpers - already in OpenAI format!
            messages = build_messages(input_data)
            tools = build_tools(input_data.tools)

            # Log prepared request
           
            # Prepare completion parameters
            completion_params = {
                "model": DEFAULT_MODEL,
                "messages": messages,
                "temperature": input_data.temperature,
                "top_p": input_data.top_p,
                "stream": True,
                "extra_headers": {
                    "HTTP-Referer": "https://inference.sh",
                    "X-Title": "Inference.sh GLM-4.6 App",
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
            try:
                # Add timeout for the initial API call
                async def create_stream():
                    return self.client.chat.completions.create(**completion_params)
                
                stream = await asyncio.wait_for(create_stream(), timeout=15.0)
            except asyncio.TimeoutError:
                raise RuntimeError("OpenRouter API call timed out after 15 seconds")
            except Exception as e:
                # Handle pre-stream errors (HTTP status errors)
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

            try:
                # Get the stream iterator by awaiting the coroutine
                stream_iterator = None
                try:
                    stream_iterator = await stream
                except Exception as e:
                    # Reflect provider error details when awaiting the stream fails
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
                async for chunk in stream_iterator:
                    chunk_count += 1
                    current_time = asyncio.get_event_loop().time()
                    
                    # Check for timeout between chunks (15 seconds)
                    if current_time - last_chunk_time > 120.0:
                        raise RuntimeError("Stream timed out - no chunks received for 120 seconds")
                    
                    last_chunk_time = current_time
                    
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
                if stream_iterator is not None and hasattr(stream_iterator, 'aclose'):
                    await stream_iterator.aclose()

        except Exception as e:
            raise RuntimeError(f"OpenRouter API error: {str(e)}")

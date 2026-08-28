import os
from typing import AsyncGenerator, List, Optional, Union
from pydantic import Field

from inferencesh import BaseApp, BaseAppOutput
from inferencesh.models.llm import (
    LLMInput,
    LLMOutput,
    LLMDelta,
    ReasoningCapabilityMixin,
    ReasoningMixin,
    ToolsCapabilityMixin,
    ToolCallsMixin,
    ImageCapabilityMixin,
    FileCapabilityMixin
)
from .openrouter import stream_completion

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DEFAULT_MODEL = "minimax/minimax-m3"


class AppInput(LLMInput, ReasoningCapabilityMixin, ToolsCapabilityMixin, ImageCapabilityMixin, FileCapabilityMixin):
    """OpenRouter input model with reasoning and tools support."""
    reasoning_exclude: bool = Field(default=False, description="Exclude reasoning tokens from response")
    context_size: int = Field(default=1048576, description="The context size for the model.")


class AppOutput(ReasoningMixin, ToolCallsMixin, LLMOutput, BaseAppOutput):
    """OpenRouter output model with reasoning, tool calls, and usage information."""
    images: Optional[List[str]] = None


class App(BaseApp):

    async def setup(self, metadata):
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")
        print("OpenRouter ready")

    async def run(self, input_data: AppInput, metadata) -> AsyncGenerator[Union[LLMDelta, AppOutput], None]:
        prev_response = ""
        prev_reasoning = ""
        last_output = None

        async for output in stream_completion(OPENROUTER_API_KEY, input_data, DEFAULT_MODEL):
            response = output.get("response", "")
            reasoning = output.get("reasoning", "")

            response_delta = response[len(prev_response):]
            reasoning_delta = reasoning[len(prev_reasoning):] if reasoning else None

            if response_delta or reasoning_delta:
                yield LLMDelta(
                    response=response_delta,
                    reasoning=reasoning_delta if reasoning_delta else None,
                )

            prev_response = response
            prev_reasoning = reasoning or ""
            last_output = output

        if last_output:
            yield AppOutput(**last_output)

    async def unload(self):
        pass

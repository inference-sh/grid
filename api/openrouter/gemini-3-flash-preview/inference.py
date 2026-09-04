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
from inferencesh.openai import OpenAIChatMixin
from .openrouter import stream_completion

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DEFAULT_MODEL = "google/gemini-3-flash-preview"


class AppInput(LLMInput, ReasoningCapabilityMixin, ToolsCapabilityMixin, ImageCapabilityMixin, FileCapabilityMixin):
    """OpenRouter input model with reasoning and tools support."""
    reasoning_exclude: bool = Field(default=False, description="Exclude reasoning tokens from response")
    context_size: int = Field(default=1048576, description="The context size for the model.")
    # Google AI Studio defaults
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_k: int = Field(default=40, ge=-1, description="Top-k sampling. -1 to disable.")
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)


class AppOutput(ReasoningMixin, ToolCallsMixin, LLMOutput, BaseAppOutput):
    """OpenRouter output model with reasoning, tool calls, and usage information."""
    images: Optional[List[str]] = None


class App(OpenAIChatMixin, BaseApp):

    async def setup(self):
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")
        print("OpenRouter ready")

    async def run(self, input_data: AppInput) -> AsyncGenerator[Union[LLMDelta, AppOutput], None]:
        last_output = None
        async for output, delta in stream_completion(OPENROUTER_API_KEY, input_data, DEFAULT_MODEL, with_deltas=True):
            if delta:
                yield LLMDelta(**delta)
            last_output = output
        if last_output:
            yield AppOutput(**last_output)

    async def unload(self):
        pass

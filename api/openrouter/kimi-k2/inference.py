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
)
from inferencesh.openai import OpenAIChatMixin
from .openrouter import stream_completion

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Load model name from model.py if exists
model_py_path = os.path.join(os.path.dirname(__file__), "model.py")
if os.path.exists(model_py_path):
    with open(model_py_path, "r") as f:
        DEFAULT_MODEL = f.read().strip()
else:
    DEFAULT_MODEL = "moonshotai/kimi-k2"


class AppInput(LLMInput, ReasoningCapabilityMixin, ToolsCapabilityMixin):
    """OpenRouter input model with reasoning and tools support."""
    reasoning_exclude: bool = Field(default=False, description="Exclude reasoning tokens from response")
    context_size: int = Field(default=200000, description="The context size for the model.")
    # Moonshot recommended for reasoning models
    temperature: float = Field(default=0.6, ge=0.0, le=2.0)


class AppOutput(ReasoningMixin, ToolCallsMixin, LLMOutput, BaseAppOutput):
    """OpenRouter output model with reasoning, tool calls, and usage information."""
    images: Optional[List[str]] = None


class App(OpenAIChatMixin, BaseApp):
    async def setup(self):
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")
        print(f"OpenRouter ready, model={DEFAULT_MODEL}")

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

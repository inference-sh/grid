import os
from typing import AsyncGenerator, List, Optional
from pydantic import Field

from inferencesh import BaseApp, BaseAppOutput
from inferencesh.models.llm import (
    LLMInput,
    LLMOutput,
    ReasoningCapabilityMixin,
    ReasoningMixin,
    ToolsCapabilityMixin,
    ToolCallsMixin
)
from .openrouter import stream_completion

# OpenRouter configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

DEFAULT_MODEL = "qwen/qwen3-8b"
class AppInput(LLMInput, ReasoningCapabilityMixin, ToolsCapabilityMixin):
    """OpenRouter input model with reasoning and tools support."""
    reasoning_exclude: bool = Field(default=False, description="Exclude reasoning tokens from response")
    context_size: int = Field(default=131072, description="The context size for the model.")
    # Qwen3 model-card recommended sampling (thinking mode defaults).
    # Non-thinking mode adjusted automatically by the qwen hook.
    # Source: https://huggingface.co/Qwen/Qwen3-8B
    temperature: float = Field(default=0.6, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    top_k: int = Field(default=20, ge=-1, description="Top-k sampling. -1 to disable.")
    min_p: float = Field(default=0.0, ge=0.0, le=1.0, description="Min-p sampling threshold.")
class AppOutput(ReasoningMixin, ToolCallsMixin, LLMOutput, BaseAppOutput):
    """OpenRouter output model with reasoning, tool calls, and usage information."""
    images: Optional[List[str]] = None
class App(BaseApp):

    async def setup(self, metadata):
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")
        print("OpenRouter ready")

    async def run(self, input_data: AppInput, metadata) -> AsyncGenerator[AppOutput, None]:

        async for output in stream_completion(OPENROUTER_API_KEY, input_data, DEFAULT_MODEL):
            yield AppOutput(**output)

    async def unload(self):
        pass

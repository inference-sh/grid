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
    ToolCallsMixin,
)
from .openrouter import stream_completion, complete
from openai import AsyncOpenAI

# Configuration
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DEFAULT_MODEL = "anthropic/claude-haiku-4.5"


class AppInput(LLMInput, ReasoningCapabilityMixin, ToolsCapabilityMixin):
    """OpenRouter input model with reasoning and tools support."""
    reasoning_exclude: bool = Field(default=False, description="Exclude reasoning tokens from response")
    context_size: int = Field(default=200000, description="The context size for the model.")
    stream: bool = Field(default=True, description="Stream the response (True) or return complete response (False)")


class AppOutput(ReasoningMixin, ToolCallsMixin, LLMOutput, BaseAppOutput):
    """OpenRouter output model with reasoning, tool calls, and usage information."""
    images: Optional[List[str]] = None


class App(BaseApp):
    def __init__(self):
        super().__init__()
        self.client = None

    async def setup(self, metadata):
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")
        self.client = AsyncOpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)
        print("OpenRouter client initialization complete!")

    async def run(self, input_data: AppInput, metadata) -> AsyncGenerator[AppOutput, None]:
        if not self.client:
            raise RuntimeError("OpenRouter client not initialized. Call setup() first.")
        
        if input_data.stream:
            async for output in stream_completion(self.client, input_data, DEFAULT_MODEL):
                yield AppOutput(**output)
        else:
            output = await complete(self.client, input_data, DEFAULT_MODEL)
            yield AppOutput(**output)

    async def unload(self):
        self.client = None

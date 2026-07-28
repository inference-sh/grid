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
    ImageCapabilityMixin,
)
from .anthropic_helper import stream_completion, complete
from anthropic import AsyncAnthropic

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_KEY")
DEFAULT_MODEL = "claude-opus-4-5-20251101"


class AppInput(LLMInput, ReasoningCapabilityMixin, ToolsCapabilityMixin, ImageCapabilityMixin):
    context_size: int = Field(default=200000, description="The context size for the model.")
    stream: bool = Field(default=True, description="Stream the response or return complete response")


class AppOutput(ReasoningMixin, ToolCallsMixin, LLMOutput, BaseAppOutput):
    pass


class App(BaseApp):
    def __init__(self):
        super().__init__()
        self.client = None

    async def setup(self):
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_KEY environment variable is required")
        self.client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        print("Anthropic client initialized for claude-opus-4-5")

    async def run(self, input_data: AppInput) -> AsyncGenerator[AppOutput, None]:
        if not self.client:
            raise RuntimeError("Anthropic client not initialized. Call setup() first.")

        if input_data.stream:
            async for output in stream_completion(self.client, input_data, DEFAULT_MODEL):
                yield AppOutput(**output)
        else:
            output = await complete(self.client, input_data, DEFAULT_MODEL)
            yield AppOutput(**output)

    async def unload(self):
        self.client = None

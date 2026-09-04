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
)
from inferencesh.openai import OpenAIChatMixin
from .anthropic_helper import stream_completion, complete
from anthropic import AsyncAnthropic

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_KEY")
DEFAULT_MODEL = "claude-mythos-5"


class AppInput(LLMInput, ReasoningCapabilityMixin, ToolsCapabilityMixin, ImageCapabilityMixin):
    context_size: int = Field(default=1000000, description="The context size for the model.")
    stream: bool = Field(default=True, description="Stream the response or return complete response")


class AppOutput(ReasoningMixin, ToolCallsMixin, LLMOutput, BaseAppOutput):
    pass


class App(OpenAIChatMixin, BaseApp):
    def __init__(self):
        super().__init__()
        self.client = None

    async def setup(self):
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_KEY environment variable is required")
        self.client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        print("Anthropic client initialized for claude-mythos-5")

    async def run(self, input_data: AppInput) -> AsyncGenerator[Union[LLMDelta, AppOutput], None]:
        if not self.client:
            raise RuntimeError("Anthropic client not initialized. Call setup() first.")

        if input_data.stream:
            last_output = None
            async for output, delta in stream_completion(self.client, input_data, DEFAULT_MODEL, max_tokens=128000, with_deltas=True):
                if delta:
                    yield LLMDelta(**delta)
                last_output = output
            if last_output:
                yield AppOutput(**last_output)
        else:
            output = await complete(self.client, input_data, DEFAULT_MODEL, max_tokens=128000)
            yield AppOutput(**output)

    async def unload(self):
        self.client = None

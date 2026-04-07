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
    FileCapabilityMixin
)
from .opper import stream_completion, complete
from openai import AsyncOpenAI

# Configuration
OPPER_BASE_URL = "https://api.opper.ai/v3/compat"
OPPER_API_KEY = os.getenv("OPPER_KEY")
DEFAULT_MODEL = "anthropic/claude-sonnet-4-6"


class AppInput(LLMInput, ReasoningCapabilityMixin, ToolsCapabilityMixin, ImageCapabilityMixin, FileCapabilityMixin):
    """Opper input model with reasoning and tools support."""
    reasoning_exclude: bool = Field(default=False, description="Exclude reasoning tokens from response")
    context_size: int = Field(default=200000, description="The context size for the model.")
    stream: bool = Field(default=True, description="Stream the response (True) or return complete response (False)")


class AppOutput(ReasoningMixin, ToolCallsMixin, LLMOutput, BaseAppOutput):
    """Opper output model with reasoning, tool calls, and usage information."""
    pass


class App(BaseApp):
    def __init__(self):
        super().__init__()
        self.client = None

    async def setup(self, metadata):
        if not OPPER_API_KEY:
            raise ValueError("OPPER_KEY environment variable is required")
        self.client = AsyncOpenAI(base_url=OPPER_BASE_URL, api_key=OPPER_API_KEY)
        print("Opper client initialization complete!")

    async def run(self, input_data: AppInput, metadata) -> AsyncGenerator[AppOutput, None]:
        if not self.client:
            raise RuntimeError("Opper client not initialized. Call setup() first.")

        print(f"Calling Opper API with model {DEFAULT_MODEL}, stream={input_data.stream}")

        if input_data.stream:
            async for output in stream_completion(self.client, input_data, DEFAULT_MODEL):
                yield AppOutput(**output)
        else:
            output = await complete(self.client, input_data, DEFAULT_MODEL)
            yield AppOutput(**output)

        print("Opper API call complete")

    async def unload(self):
        self.client = None

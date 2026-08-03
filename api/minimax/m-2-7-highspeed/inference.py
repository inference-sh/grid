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
from .minimax_llm import stream_completion

MINIMAX_KEY = os.getenv("MINIMAX_KEY")
DEFAULT_MODEL = "MiniMax-M2.7-highspeed"


class AppInput(LLMInput, ReasoningCapabilityMixin, ToolsCapabilityMixin, ImageCapabilityMixin, FileCapabilityMixin):
    reasoning_exclude: bool = Field(default=False, description="Exclude reasoning tokens from response")
    context_size: int = Field(default=204800, description="The context size for the model.")


class AppOutput(ReasoningMixin, ToolCallsMixin, LLMOutput, BaseAppOutput):
    images: Optional[List[str]] = None


class App(BaseApp):

    async def setup(self, metadata):
        if not MINIMAX_KEY:
            raise ValueError("MINIMAX_KEY environment variable is required")
        print("MiniMax M2.7-highspeed ready")

    async def run(self, input_data: AppInput, metadata) -> AsyncGenerator[AppOutput, None]:

        async for output in stream_completion(MINIMAX_KEY, input_data, DEFAULT_MODEL):
            yield AppOutput(**output)

    async def unload(self):
        pass

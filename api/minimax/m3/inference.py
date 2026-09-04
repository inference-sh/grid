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
from .minimax_llm import stream_completion

MINIMAX_KEY = os.getenv("MINIMAX_KEY")
DEFAULT_MODEL = "MiniMax-M3"


class AppInput(LLMInput, ReasoningCapabilityMixin, ToolsCapabilityMixin, ImageCapabilityMixin, FileCapabilityMixin):
    reasoning_exclude: bool = Field(default=False, description="Exclude reasoning tokens from response")
    context_size: int = Field(default=1048576, description="The context size for the model.")


class AppOutput(ReasoningMixin, ToolCallsMixin, LLMOutput, BaseAppOutput):
    images: Optional[List[str]] = None


class App(OpenAIChatMixin, BaseApp):

    async def setup(self):
        if not MINIMAX_KEY:
            raise ValueError("MINIMAX_KEY environment variable is required")
        print("MiniMax M3 ready")

    async def run(self, input_data: AppInput) -> AsyncGenerator[Union[LLMDelta, AppOutput], None]:
        last_output = None
        async for output, delta in stream_completion(MINIMAX_KEY, input_data, DEFAULT_MODEL, with_deltas=True):
            if delta:
                yield LLMDelta(**delta)
            last_output = output
        if last_output:
            yield AppOutput(**last_output)

    async def unload(self):
        pass

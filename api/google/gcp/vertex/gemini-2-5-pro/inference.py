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
from .gemini_chat_helper import create_vertex_client, setup_logger, stream_completion

DEFAULT_MODEL = "gemini-2.5-pro"


class AppInput(LLMInput, ReasoningCapabilityMixin, ToolsCapabilityMixin, ImageCapabilityMixin):
    context_size: int = Field(default=1048576, description="The context size for the model.")
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_k: int = Field(default=40, ge=-1, description="Top-k sampling. -1 to disable.")
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)


class AppOutput(ReasoningMixin, ToolCallsMixin, LLMOutput, BaseAppOutput):
    pass


class App(OpenAIChatMixin, BaseApp):

    async def setup(self):
        self.logger = setup_logger(__name__)
        self.client = create_vertex_client()
        self.logger.info("Gemini 2.5 Pro (Vertex AI) ready")

    async def run(self, input_data: AppInput) -> AsyncGenerator[Union[LLMDelta, AppOutput], None]:
        last_output = None
        async for output, delta in stream_completion(self.client, input_data, DEFAULT_MODEL, can_disable_thinking=False, with_deltas=True):
            if delta:
                yield LLMDelta(**delta)
            last_output = output
        if last_output:
            yield AppOutput(**last_output)

    async def unload(self):
        pass

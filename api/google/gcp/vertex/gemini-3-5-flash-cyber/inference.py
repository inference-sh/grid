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
from .gemini_chat_helper import create_vertex_client, setup_logger, stream_completion

DEFAULT_MODEL = "gemini-3.5-flash-cyber"


class AppInput(LLMInput, ReasoningCapabilityMixin, ToolsCapabilityMixin, ImageCapabilityMixin):
    context_size: int = Field(default=1048576, description="The context size for the model.")
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_k: int = Field(default=40, ge=-1, description="Top-k sampling. -1 to disable.")
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)


class AppOutput(ReasoningMixin, ToolCallsMixin, LLMOutput, BaseAppOutput):
    pass


class App(BaseApp):

    async def setup(self):
        self.logger = setup_logger(__name__)
        self.client = create_vertex_client()
        self.logger.info("Gemini 3.5 Flash Cyber (Vertex AI) ready")

    async def run(self, input_data: AppInput) -> AsyncGenerator[AppOutput, None]:
        async for output in stream_completion(self.client, input_data, DEFAULT_MODEL):
            yield AppOutput(**output)

    async def unload(self):
        pass

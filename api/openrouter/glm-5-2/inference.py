import os
from typing import AsyncGenerator, List, Optional
from pydantic import Field

from inferencesh import BaseApp, BaseAppOutput
from inferencesh.models.llm import (
    LLMInput,
    LLMOutput,
    ToolsCapabilityMixin,
    ToolCallsMixin
)
from .openrouter import stream_completion

# OpenRouter configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

DEFAULT_MODEL = "z-ai/glm-5.2"

class AppInput(LLMInput, ToolsCapabilityMixin):
    """OpenRouter input model for GLM 5.2."""
    context_size: int = Field(default=1048576, description="The context size for the model.")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=-1, ge=-1, description="Top-k sampling. -1 to disable.")
    min_p: float = Field(default=0.0, ge=0.0, le=1.0, description="Min-p sampling threshold.")

class AppOutput(ToolCallsMixin, LLMOutput, BaseAppOutput):
    """OpenRouter output model with tool calls and usage information."""
    images: Optional[List[str]] = None

class App(BaseApp):

    async def setup(self, metadata):
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")
        print("OpenRouter ready")

    async def run(self, input_data: AppInput, metadata) -> AsyncGenerator[AppOutput, None]:
        print(f"Calling OpenRouter model={DEFAULT_MODEL}")

        async for output in stream_completion(OPENROUTER_API_KEY, input_data, DEFAULT_MODEL):
            yield AppOutput(**output)

    async def unload(self):
        pass

from typing import AsyncGenerator, Union
from pydantic import Field

from inferencesh.models.llm import LLMDelta, LLMInput
from .openrouter import OpenRouterChatApp, OpenRouterOutput

DEFAULT_MODEL = "google/gemini-3-pro-image-preview"


class AppInput(LLMInput):
    """OpenRouter input model with reasoning and tools support."""
    reasoning_exclude: bool = Field(default=False, description="Exclude reasoning tokens from response")
    context_size: int = Field(default=200000, description="The context size for the model.")
    # Google AI Studio defaults
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_k: int = Field(default=40, ge=-1, description="Top-k sampling. -1 to disable.")
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)


class AppOutput(OpenRouterOutput):
    """OpenRouter output model with reasoning, tool calls, and usage information."""


class App(OpenRouterChatApp):
    async def run(self, input_data: AppInput) -> AsyncGenerator[Union[LLMDelta, AppOutput], None]:
        async for out in self._stream(input_data, AppOutput, DEFAULT_MODEL):
            yield out

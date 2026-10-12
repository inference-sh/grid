from typing import AsyncGenerator, Union
from pydantic import Field

from inferencesh.models.llm import LLMDelta, LLMInput
from .openrouter import OpenRouterChatApp, OpenRouterOutput

# Kimi K2 Instruct (0905), the non-thinking K2; openrouter/kimi-k2-thinking serves the thinking model.
DEFAULT_MODEL = "moonshotai/kimi-k2-0905"


class AppInput(LLMInput):
    """OpenRouter input model with reasoning and tools support."""
    reasoning_exclude: bool = Field(default=False, description="Exclude reasoning tokens from response")
    context_size: int = Field(default=262144, description="The context size for the model.")
    # Moonshot recommended for K2
    temperature: float = Field(default=0.6, ge=0.0, le=2.0)


class AppOutput(OpenRouterOutput):
    """OpenRouter output model with reasoning, tool calls, and usage information."""


class App(OpenRouterChatApp):
    async def run(self, input_data: AppInput) -> AsyncGenerator[Union[LLMDelta, AppOutput], None]:
        async for out in self._stream(input_data, AppOutput, DEFAULT_MODEL):
            yield out

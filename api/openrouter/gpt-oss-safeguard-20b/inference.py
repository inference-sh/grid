from typing import AsyncGenerator, Literal, Union
from pydantic import Field

from inferencesh.models.llm import LLMDelta, LLMInput
from .openrouter import OpenRouterChatApp, OpenRouterOutput

DEFAULT_MODEL = "openai/gpt-oss-safeguard-20b"


class AppInput(LLMInput):
    """OpenRouter input model with reasoning and tools support."""
    # Reasoning is mandatory on this endpoint: OpenRouter refuses effort "none".
    reasoning_effort: Literal["low", "medium", "high"] = Field(default="low", description="Reasoning effort. This model always reasons; it cannot be turned off.")
    reasoning_exclude: bool = Field(default=False, description="Exclude reasoning tokens from response")
    context_size: int = Field(default=200000, description="The context size for the model.")


class AppOutput(OpenRouterOutput):
    """OpenRouter output model with reasoning, tool calls, and usage information."""


class App(OpenRouterChatApp):
    async def run(self, input_data: AppInput) -> AsyncGenerator[Union[LLMDelta, AppOutput], None]:
        async for out in self._stream(input_data, AppOutput, DEFAULT_MODEL):
            yield out

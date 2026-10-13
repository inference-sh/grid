from typing import AsyncGenerator, Union
from pydantic import Field

from inferencesh.models.llm import LLMDelta, LLMInput
from .openrouter import OpenRouterChatApp, OpenRouterOutput


class AppInput(LLMInput):
    """OpenRouter input model with reasoning and tools support."""
    model: str = Field(default="openai/gpt-6-luna", description="The OpenRouter model id to use for the inference, e.g. openai/gpt-6-luna or anthropic/claude-sonnet-5.5.")


class AppOutput(OpenRouterOutput):
    """OpenRouter output model with reasoning, tool calls, and usage information."""


class App(OpenRouterChatApp):
    async def run(self, input_data: AppInput) -> AsyncGenerator[Union[LLMDelta, AppOutput], None]:
        async for out in self._stream(input_data, AppOutput, input_data.model):
            yield out

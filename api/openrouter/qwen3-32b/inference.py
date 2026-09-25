from typing import AsyncGenerator, Union
from pydantic import Field

from inferencesh.models.llm import LLMDelta, LLMInput
from .openrouter import OpenRouterChatApp, OpenRouterOutput

DEFAULT_MODEL = "qwen/qwen3-32b"


class AppInput(LLMInput):
    """OpenRouter input model with reasoning and tools support."""
    reasoning_exclude: bool = Field(default=False, description="Exclude reasoning tokens from response")
    context_size: int = Field(default=131072, description="The context size for the model.")
    # Qwen3 model-card recommended sampling (thinking mode defaults).
    # Non-thinking mode adjusted automatically by the qwen hook.
    # Source: https://huggingface.co/Qwen/Qwen3-32B
    temperature: float = Field(default=0.6, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    top_k: int = Field(default=20, ge=-1, description="Top-k sampling. -1 to disable.")
    min_p: float = Field(default=0.0, ge=0.0, le=1.0, description="Min-p sampling threshold.")


class AppOutput(OpenRouterOutput):
    """OpenRouter output model with reasoning, tool calls, and usage information."""


class App(OpenRouterChatApp):
    async def run(self, input_data: AppInput) -> AsyncGenerator[Union[LLMDelta, AppOutput], None]:
        async for out in self._stream(input_data, AppOutput, DEFAULT_MODEL):
            yield out

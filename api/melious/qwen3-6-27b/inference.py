from typing import AsyncGenerator, Union
from pydantic import Field

from inferencesh import BaseApp
from inferencesh.models.llm import LLMInput, LLMOutput, LLMDelta
from inferencesh.openai import OpenAIChatMixin
from .melious import stream_completion

DEFAULT_MODEL = "qwen3.6-27b"
VISION = True


class AppInput(LLMInput):
    """Qwen 3.6 27B via the Melious Chat Completions API: text and image input, tools, reasoning."""
    context_size: int = Field(default=262000, description="The context size for the model.")


class AppOutput(LLMOutput):
    """Response text, reasoning, tool calls and token usage."""
    pass


class App(OpenAIChatMixin, BaseApp):

    async def setup(self, metadata):
        print(f"Melious ready model={DEFAULT_MODEL}")

    async def run(self, input_data: AppInput) -> AsyncGenerator[Union[LLMDelta, AppOutput], None]:
        last_output = None

        async for output, delta in stream_completion(input_data, DEFAULT_MODEL, vision=VISION, with_deltas=True):
            if delta:
                yield LLMDelta(**delta)
            last_output = output

        if last_output:
            yield AppOutput(**last_output)

    async def unload(self):
        pass

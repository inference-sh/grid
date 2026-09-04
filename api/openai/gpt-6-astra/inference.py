import os
from typing import AsyncGenerator, Union
from pydantic import Field

from inferencesh import BaseApp
from inferencesh.models.llm import LLMInput, LLMOutput, LLMDelta
from inferencesh.openai import OpenAIChatMixin
from .openai_llm import stream_completion

OPENAI_KEY = os.getenv("OPENAI_KEY")
MODEL = "gpt-6-astra"
MAX_OUTPUT_TOKENS = 128000
# Whether the model accepts reasoning effort "none". When False, "none" is sent as "low".
SUPPORTS_NONE_REASONING = False


class AppInput(LLMInput):
    """gpt-6-astra via the OpenAI Responses API: text, image and file input, tools, reasoning."""
    context_size: int = Field(default=1050000, description="The context size for the model.")


class AppOutput(LLMOutput):
    """Response text, reasoning summary, tool calls and token usage."""
    pass


class App(OpenAIChatMixin, BaseApp):

    async def setup(self):
        if not OPENAI_KEY:
            raise ValueError("OPENAI_KEY environment variable is required")
        print(f"OpenAI ready model={MODEL}")

    async def run(self, input_data: AppInput) -> AsyncGenerator[Union[LLMDelta, AppOutput], None]:
        last_output = None

        async for output, delta in stream_completion(
            OPENAI_KEY,
            input_data,
            MODEL,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            supports_none_reasoning=SUPPORTS_NONE_REASONING,
            with_deltas=True,
        ):
            if delta:
                yield LLMDelta(**delta)
            last_output = output

        if last_output:
            yield AppOutput(**last_output)

    async def unload(self):
        pass

from typing import AsyncGenerator, Optional, Union
from pydantic import Field

from inferencesh import BaseApp
from inferencesh.models.llm import LLMInput, LLMOutput, LLMDelta
from inferencesh.openai import OpenAIChatMixin
from .xai_llm import stream_completion

DEFAULT_MODEL = "grok-build-0.1"
MAX_OUTPUT_TOKENS = 128000
# xAI publishes no effort levels for this model, so reasoning_effort is not sent.
REASONING_EFFORTS = None


class AppInput(LLMInput):
    """grok-build-0.1 via the xAI Responses API: text and image input, tools, reasoning."""
    context_size: int = Field(default=256000, description="The context size for the model.")


class AppOutput(LLMOutput):
    """Response text, reasoning, tool calls and token usage."""
    notice: Optional[str] = Field(default=None, description="Set when xAI rejected the request under its usage guidelines.")


class App(OpenAIChatMixin, BaseApp):

    async def setup(self, metadata):
        print(f"xAI ready model={DEFAULT_MODEL}")

    async def run(self, input_data: AppInput) -> AsyncGenerator[Union[LLMDelta, AppOutput], None]:
        last_output = None

        async for output, delta in stream_completion(
            input_data,
            DEFAULT_MODEL,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            efforts=REASONING_EFFORTS,
            with_deltas=True,
        ):
            if delta:
                yield LLMDelta(**delta)
            last_output = output

        if last_output:
            yield AppOutput(**last_output)

    async def unload(self):
        pass

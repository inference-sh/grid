from typing import AsyncGenerator, Union
from pydantic import Field

from inferencesh import BaseApp
from inferencesh.models.llm import LLMInput, LLMOutput, LLMDelta
from inferencesh.openai import OpenAIChatMixin
from .meta_llm import stream_completion

MODEL = "muse-wonder-1.0"
# The largest max_output_tokens accepted; sent only when the caller sets max_tokens.
MAX_OUTPUT_TOKENS = 131072
# Reasoning cannot be turned off: "none" is sent as "minimal".
REASONING_EFFORTS = ("minimal", "low", "medium", "high", "xhigh", "max")


class AppInput(LLMInput):
    """muse-wonder-1.0 via the Meta Model API (Responses): text, image, video, audio and PDF input, tools, reasoning."""
    context_size: int = Field(default=1048576, description="The context size for the model.")


class AppOutput(LLMOutput):
    """Response text, reasoning summary, tool calls and token usage."""
    pass


class App(OpenAIChatMixin, BaseApp):

    async def setup(self, metadata):
        print(f"Meta ready model={MODEL}")

    async def run(self, input_data: AppInput) -> AsyncGenerator[Union[LLMDelta, AppOutput], None]:
        last_output = None

        async for output, delta in stream_completion(
            input_data,
            MODEL,
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

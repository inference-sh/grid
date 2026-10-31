import os
from typing import AsyncGenerator, Union
from pydantic import Field

from inferencesh import BaseApp
from inferencesh.models.llm import LLMInput, LLMOutput, LLMDelta
from inferencesh.openai import OpenAIChatMixin
from .openai_llm import stream_completion

# Key from the workspace's own Daybreak-enabled OpenAI project, not the platform OPENAI_KEY.
OPENAI_DAYBREAK_KEY = os.getenv("OPENAI_DAYBREAK_KEY")
MODEL = "gpt-5.6-cyber"
MAX_OUTPUT_TOKENS = 128000
# Whether the model accepts reasoning effort "none". When False, "none" is sent as "low".
SUPPORTS_NONE_REASONING = False
# The program value OpenAI expects for this model (not the org's approval level).
ACCESS_PROGRAMS = {"cyber": "daybreak_red"}
# Shown in front of OpenAI's error when the project lacks this access.
ACCESS_HINT = "GPT-5.6 Cyber needs Daybreak Red approval plus OpenAI's separate approval for this model."


class AppInput(LLMInput):
    """gpt-5.6-cyber via the OpenAI Responses API with Daybreak (daybreak_red): text and image input, tools, reasoning."""
    context_size: int = Field(default=400000, description="The context size for the model.")


class AppOutput(LLMOutput):
    """Response text, reasoning summary, tool calls and token usage."""
    pass


class App(OpenAIChatMixin, BaseApp):

    async def setup(self):
        if not OPENAI_DAYBREAK_KEY:
            raise ValueError("OPENAI_DAYBREAK_KEY environment variable is required")
        print(f"OpenAI Daybreak ready model={MODEL} access_programs={ACCESS_PROGRAMS}")

    async def run(self, input_data: AppInput) -> AsyncGenerator[Union[LLMDelta, AppOutput], None]:
        last_output = None

        async for output, delta in stream_completion(
            OPENAI_DAYBREAK_KEY,
            input_data,
            MODEL,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            supports_none_reasoning=SUPPORTS_NONE_REASONING,
            with_deltas=True,
            access_programs=ACCESS_PROGRAMS,
            access_hint=ACCESS_HINT,
        ):
            if delta:
                yield LLMDelta(**delta)
            last_output = output

        if last_output:
            yield AppOutput(**last_output)

    async def unload(self):
        pass

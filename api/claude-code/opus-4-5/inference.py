from typing import AsyncGenerator, Union
from pydantic import Field

from inferencesh import BaseApp, BaseAppOutput
from inferencesh.models.llm import LLMInput, LLMOutput, LLMDelta
from inferencesh.openai import OpenAIChatMixin
from .claude_code_helper import Serve, UsageMixin, ensure_claude, find_belt, stream_run

DEFAULT_MODEL = "claude-opus-4-5"


class AppInput(LLMInput):
    context_size: int = Field(default=1000000, description="The context size for the model.")


class AppOutput(LLMOutput, BaseAppOutput):
    pass


class App(UsageMixin, OpenAIChatMixin, BaseApp):
    def __init__(self):
        super().__init__()
        self.serve = None

    async def setup(self):
        self.serve = Serve(find_belt(), ensure_claude())
        await self.serve.start()
        print(f"belt harness serve ready for {DEFAULT_MODEL}")

    async def run(self, input_data: AppInput) -> AsyncGenerator[Union[LLMDelta, AppOutput], None]:
        if not self.serve:
            raise RuntimeError("belt harness serve not started. Call setup() first.")
        context = getattr(self, "context", None)
        metadata = getattr(context, "metadata", None)
        rid = getattr(metadata, "task_id", None) or "local"
        async for kind, data in stream_run(self.serve, rid, DEFAULT_MODEL, input_data):
            if kind == "delta":
                yield LLMDelta(**data)
            else:
                yield AppOutput(**data)

    async def on_cancel(self):
        # The kernel calls this on a cancel; the run then ends with the text
        # streamed so far, and serve stops the claude call.
        if self.serve:
            self.serve.cancel_all()
        return True

    async def unload(self):
        if self.serve:
            await self.serve.close()
        self.serve = None

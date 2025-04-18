import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput
from pydantic import Field, BaseModel
from typing import AsyncGenerator, List, Optional
from queue import Queue
from threading import Thread
import asyncio
from llama_cpp import Llama

class ContextMessage(BaseModel):
    role: str = Field(
        description="The role of the message",
        enum=["user", "assistant", "system"]
    )
    content: str = Field(
        description="The content of the message"
    )

class AppInput(BaseAppInput):
    system_prompt: str = Field(
        description="The system prompt to use for the model",
        default="You are a helpful assistant that can answer questions and help with tasks.",
        examples=[
            "You are a helpful assistant that can answer questions and help with tasks.",
            "You are a certified medical professional who can provide accurate health information.",
            "You are a certified financial advisor who can give sound investment guidance.",
            "You are a certified cybersecurity expert who can explain security best practices.",
            "You are a certified environmental scientist who can discuss climate and sustainability.",
        ]
    )
    context: List[ContextMessage] = Field(
        description="The context to use for the model",
        default=[],
        examples=[
            [
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "The capital of France is Paris."}
            ]
        ]
    )
    user_prompt: str = Field(
        description="The user prompt to use for the model",
        examples=[
            "What is the capital of France?",
            "Can you help me write a poem about spring?"
        ],
    )

class AppOutput(BaseAppOutput):
    response: str

class App(BaseApp):
    async def setup(self):
        # Initialize llama.cpp model
        self.model = Llama.from_pretrained(
            repo_id="google/gemma-3-27b-it-qat-q4_0-gguf",
            filename="gemma-3-27b-it-q4_0.gguf",
            verbose=True,
            n_gpu_layers=-1,
            n_ctx=4096
        )

    async def run(self, input_data: AppInput) -> AsyncGenerator[AppOutput, None]:
        messages = [
            {"role": "system", "content": input_data.system_prompt},
            *input_data.context,
            {"role": "user", "content": input_data.user_prompt}
        ]

        response_queue: "Queue[Optional[str]]" = Queue()

        def generation_thread():
            try:
                for chunk in self.model.create_chat_completion(
                    messages=messages,
                    stream=True,
                    temperature=0.7,
                    top_p=0.95,
                    max_tokens=512
                ):
                    delta = chunk.get("choices", [{}])[0] \
                                  .get("delta", {}) \
                                  .get("content", "")
                    if delta:
                        response_queue.put(delta)
            finally:
                response_queue.put(None)

        thread = Thread(target=generation_thread, daemon=True)
        thread.start()

        buffer = ""
        loop = asyncio.get_event_loop()
        while True:
            piece = await loop.run_in_executor(None, response_queue.get)
            if piece is None:
                break
            buffer += piece
            yield AppOutput(response=buffer)

        thread.join()

    async def unload(self):
        del self.model
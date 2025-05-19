import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, LLMInput, ContextMessage
from pydantic import Field, BaseModel
from typing import AsyncGenerator, List, Optional
from queue import Queue
from threading import Thread
import asyncio
from llama_cpp import Llama

class AppInput(LLMInput):
    pass

class AppOutput(BaseAppOutput):
    response: str
    thinking_content: Optional[str] = None

class App(BaseApp):
    async def setup(self):
        # Initialize llama.cpp model
        self.model = Llama.from_pretrained(
            repo_id="Qwen/QwQ-32B-GGUF",
            filename="qwq-32b-q4_k_m.gguf",
            verbose=True,
            n_gpu_layers=-1,
            n_ctx=4096
        )

    async def run(self, input_data: AppInput) -> AsyncGenerator[AppOutput, None]:
        messages = [
            {"role": "system", "content": input_data.system_prompt},
            *input_data.context,
            {"role": "user", "content": input_data.text}
        ]

        response_queue: "Queue[Optional[str]]" = Queue()

        def generation_thread():
            try:
                for chunk in self.model.create_chat_completion(
                    messages=messages,
                    stream=True,
                    temperature=0.7,
                    top_p=0.95,
                    max_tokens=2048
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
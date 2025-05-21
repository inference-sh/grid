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
    enable_thinking: bool = Field(
        description="Whether to enable thinking mode for complex reasoning",
        default=True
    )

class AppOutput(BaseAppOutput):
    response: str
    thinking_content: Optional[str] = None

class App(BaseApp):
    async def setup(self, metadata):
        # Initialize llama.cpp model
        self.model = Llama.from_pretrained(
            repo_id="unsloth/Qwen3-30B-A3B-GGUF",
            filename="Qwen3-30B-A3B-Q6_K.gguf",
            verbose=True,
            n_gpu_layers=-1,
            n_ctx=32768
        )

    async def run(self, input_data: AppInput) -> AsyncGenerator[AppOutput, None]:
        # Modify user prompt based on thinking mode
        user_prompt = input_data.text
        if input_data.enable_thinking:
            user_prompt = f"{user_prompt} /think"
        else:
            user_prompt = f"{user_prompt} /no_think"

        messages = [
            {"role": "system", "content": input_data.system_prompt},
            *input_data.context,
            {"role": "user", "content": user_prompt}
        ]

        response_queue: "Queue[Optional[str]]" = Queue()

        def generation_thread():
            try:
                # Adjust temperature and top_p based on thinking mode
                temp = 0.6 if input_data.enable_thinking else 0.7
                top_p = 0.95 if input_data.enable_thinking else 0.8
                
                for chunk in self.model.create_chat_completion(
                    messages=messages,
                    stream=True,
                    temperature=temp,
                    top_p=top_p,
                    max_tokens=24576
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
        thinking_content = ""
        in_thinking = input_data.enable_thinking
        
        loop = asyncio.get_event_loop()
        while True:
            piece = await loop.run_in_executor(None, response_queue.get)
            if piece is None:
                break
            print(piece)
            # Handle thinking vs response content
            if "</think>" in piece:
                parts = piece.split("</think>")
                if in_thinking:
                    thinking_content += parts[0].replace("<think>", "")
                    buffer = parts[1] if len(parts) > 1 else ""
                    in_thinking = False
                else:
                    buffer += piece
            else:
                if in_thinking:
                    thinking_content += piece.replace("<think>", "")
                else:
                    buffer += piece
                    
            yield AppOutput(
                response=buffer.strip(),
                thinking_content=thinking_content.strip() if thinking_content else None
            )

        thread.join()

    async def unload(self):
        del self.model
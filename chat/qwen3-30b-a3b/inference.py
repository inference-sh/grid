import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from inferencesh import BaseApp, BaseAppOutput, LLMInput
from pydantic import Field
from typing import AsyncGenerator, Optional
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

def Qwen3MessageBuilder(input_data: AppInput):
    messages = [
        {"role": "system", "content": input_data.system_prompt}
    ]
    # Add context messages
    for msg in input_data.context:
        content = []
        if hasattr(msg, 'text') and msg.text:
            content.append({"type": "text", "text": msg.text})
        # flatten for llama.cpp compatibility
        if len(content) == 1:
            content = content[0]["text"]
        messages.append({
            "role": msg.role,
            "content": content
        })
    # Add user message with thinking mode
    user_prompt = input_data.text
    if input_data.enable_thinking:
        user_prompt = f"{user_prompt} /think"
    else:
        user_prompt = f"{user_prompt} /no_think"
    user_content = [{"type": "text", "text": user_prompt}]
    user_content = user_content[0]["text"]
    messages.append({"role": "user", "content": user_content})
    return messages

def stream_generate(model, messages, AppOutput, enable_thinking):
    response_queue: "Queue[Optional[str]]" = Queue()
    thread_exception = None

    def generation_thread():
        nonlocal thread_exception
        try:
            temp = 0.6 if enable_thinking else 0.7
            top_p = 0.95 if enable_thinking else 0.8
            for chunk in model.create_chat_completion(
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
        except Exception as e:
            thread_exception = e
        finally:
            response_queue.put(None)

    thread = Thread(target=generation_thread, daemon=True)
    thread.start()

    buffer = ""
    thinking_content = ""
    in_thinking = enable_thinking
    try:
        while True:
            piece = response_queue.get()
            if piece is None:
                break
            if thread_exception:
                raise thread_exception
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
        if thread_exception:
            raise thread_exception
    finally:
        thread.join(timeout=2.0)

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

    async def run(self, input_data: AppInput, metadata) -> AsyncGenerator[AppOutput, None]:
        messages = Qwen3MessageBuilder(input_data)
        for output in stream_generate(self.model, messages, AppOutput, input_data.enable_thinking):
            yield output

    async def unload(self):
        del self.model
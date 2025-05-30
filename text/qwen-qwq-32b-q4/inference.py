import os
import base64
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from inferencesh import BaseApp, BaseAppOutput, LLMInput
from typing import AsyncGenerator, Optional
from threading import Thread
from queue import Queue
from llama_cpp import Llama

class AppInput(LLMInput):
    pass

class AppOutput(BaseAppOutput):
    response: str
    thinking_content: Optional[str] = None

def image_to_base64_data_uri(file_path):
    with open(file_path, "rb") as img_file:
        base64_data = base64.b64encode(img_file.read()).decode('utf-8')
        return f"data:image/png;base64,{base64_data}"

def QwenMessageBuilder(input_data: AppInput):
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": input_data.system_prompt}],
        }
    ]

    # Add context messages
    for msg in input_data.context:
        message_content = []
        if msg.text:
            message_content.append({"type": "text", "text": msg.text})
        if msg.image and msg.image.path:
            image_data_uri = image_to_base64_data_uri(msg.image.path)
            message_content.append({"type": "image_url", "image_url": {"url": image_data_uri}})
        elif msg.image and msg.image.uri:
            message_content.append({"type": "image_url", "image_url": {"url": msg.image.uri}})
        messages.append({
            "role": msg.role,
            "content": message_content
        })

    # Add user message with text and image if provided
    user_content = []
    user_text = input_data.text
    if user_text:
        user_content.append({"type": "text", "text": user_text})
    if input_data.image and input_data.image.path:
        image_data_uri = image_to_base64_data_uri(input_data.image.path)
        user_content.append({"type": "image_url", "image_url": {"url": image_data_uri}})
    elif input_data.image and input_data.image.uri:
        user_content.append({"type": "image_url", "image_url": {"url": input_data.image.uri}})
    messages.append({"role": "user", "content": user_content})

    return messages

def stream_generate(model, messages, AppOutput):
    response_queue: "Queue[Optional[str]]" = Queue()
    thread_exception = None

    def generation_thread():
        nonlocal thread_exception
        try:
            for chunk in model.create_chat_completion(
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
        except Exception as e:
            thread_exception = e
        finally:
            response_queue.put(None)

    thread = Thread(target=generation_thread, daemon=True)
    thread.start()

    buffer = ""
    thinking_content = ""
    in_thinking = True
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
        self.model = Llama.from_pretrained(
            repo_id="Qwen/QwQ-32B-GGUF",
            filename="qwq-32b-q4_k_m.gguf",
            verbose=True,
            n_gpu_layers=-1,
            n_ctx=4096
        )

    async def run(self, input_data: AppInput, metadata) -> AsyncGenerator[AppOutput, None]:
        messages = QwenMessageBuilder(input_data)
        for output in stream_generate(self.model, messages, AppOutput):
            yield output

    async def unload(self):
        del self.model
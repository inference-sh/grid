import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from inferencesh import BaseApp, BaseAppOutput, BaseAppInput
from pydantic import Field, BaseModel
from typing import AsyncGenerator, Optional
from queue import Queue
from threading import Thread
import asyncio
from llama_cpp import Llama

class ContextMessage(BaseModel):
    role: str = Field(
        description="The role of the message",
        enum=["user", "assistant", "system"]
    )
    text: str = Field(
        description="The text content of the message"
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
    context: list[ContextMessage] = Field(
        description="The context to use for the model",
        examples=[
            [
                {"role": "user", "content": [{"type": "text", "text": "What is the capital of France?"}]}, 
                {"role": "assistant", "content": [{"type": "text", "text": "The capital of France is Paris."}]}
            ],
            [
                {"role": "user", "content": [{"type": "text", "text": "What is the weather like today?"}]}, 
                {"role": "assistant", "content": [{"type": "text", "text": "I apologize, but I don't have access to real-time weather information. You would need to check a weather service or app to get current weather conditions for your location."}]}
            ],
            [
                {"role": "user", "content": [{"type": "text", "text": "Can you help me write a poem about spring?"}]}, 
                {"role": "assistant", "content": [{"type": "text", "text": "Here's a short poem about spring:\n\nGreen buds awakening,\nSoft rain gently falling down,\nNew life springs anew.\n\nWarm sun breaks through clouds,\nBirds return with joyful song,\nNature's sweet rebirth."}]}
            ],
            [
                {"role": "user", "content": [{"type": "text", "text": "Explain quantum computing in simple terms"}]}, 
                {"role": "assistant", "content": [{"type": "text", "text": "Quantum computing is like having a super-powerful calculator that can solve many problems at once instead of one at a time. While regular computers use bits (0s and 1s), quantum computers use quantum bits or \"qubits\" that can be both 0 and 1 at the same time - kind of like being in two places at once! This allows them to process huge amounts of information much faster than regular computers for certain types of problems."}]}
            ]
        ],
        default=[]
    )
    text: str = Field(
        description="The user prompt to use for the model",
        examples=[
            "What is the capital of France?",
            "What is the weather like today?",
            "Can you help me write a poem about spring?",
            "Explain quantum computing in simple terms"
        ],
    )

class AppOutput(BaseAppOutput):
    response: str

def GraniteMessageBuilder(input_data: AppInput):
    messages = [
        {
            "role": "system",
            "content": input_data.system_prompt,
        }
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
    # Add user message
    if hasattr(input_data, 'text') and input_data.text:
        user_content = [{"type": "text", "text": input_data.text}]
        user_content = user_content[0]["text"]
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
                max_tokens=512
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
    try:
        while True:
            piece = response_queue.get()
            if piece is None:
                break
            if thread_exception:
                raise thread_exception
            buffer += piece
            yield AppOutput(response=buffer)
        if thread_exception:
            raise thread_exception
    finally:
        thread.join(timeout=2.0)

class App(BaseApp):
    async def setup(self, metadata):
        self.model = Llama.from_pretrained(
            repo_id="ibm-granite/granite-8b-code-instruct-4k-GGUF",
            filename="granite-8b-code-instruct.Q4_K_M.gguf",
            verbose=True,
            n_gpu_layers=-1,
            n_ctx=4096
        )

    async def run(self, input_data: AppInput, metadata) -> AsyncGenerator[AppOutput, None]:
        messages = GraniteMessageBuilder(input_data)
        for output in stream_generate(self.model, messages, AppOutput):
            yield output

    async def unload(self):
        del self.model
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from inferencesh import BaseApp, BaseAppOutput, LLMInputWithImage, ContextMessage
from pydantic import Field, BaseModel
from typing import AsyncGenerator, List, Optional
from queue import Queue
from threading import Thread
import asyncio
import PIL
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Gemma3ChatHandler
from huggingface_hub import hf_hub_download
import os.path
import base64

class AppInput(LLMInputWithImage):
    # enable_thinking: bool = Field(
    #     description="Whether to enable thinking mode for complex reasoning",
    #     default=False
    # )
    pass

class AppOutput(BaseAppOutput):
    response: str
    # thinking_content: Optional[str] = None
    

def image_to_base64_data_uri(file_path):
    with open(file_path, "rb") as img_file:
        base64_data = base64.b64encode(img_file.read()).decode('utf-8')
        return f"data:image/png;base64,{base64_data}"

def Gemma3MessageBuilder(input_data: AppInput):
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
    print(f"input_data.image: {input_data.image}")
    if user_text:
        user_content.append({"type": "text", "text": user_text})
    if input_data.image and input_data.image.path:
        image_data_uri = image_to_base64_data_uri(input_data.image.path)
        user_content.append({"type": "image_url", "image_url": {"url": image_data_uri}})
    elif input_data.image and input_data.image.uri:
        user_content.append({"type": "image_url", "image_url": {"url": input_data.image.uri}})
    messages.append({"role": "user", "content": user_content})

    print(f"messages: {messages}")
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
                max_tokens=512,
                stop=['<end_of_turn>', '<eos>']
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
            # Clean up any special tokens that might be in the response
            piece = piece.replace("<|im_end|>", "").replace("<|im_start|>", "").replace("<end_of_turn>", "")
            buffer += piece
            yield AppOutput(response=buffer)
        if thread_exception:
            raise thread_exception
    finally:
        thread.join(timeout=2.0)

class App(BaseApp):
    async def setup(self, metadata):
        try:
            # Download CLIP model from Hugging Face Hub
            mmproj_repo = "ggml-org/gemma-3-27b-it-GGUF"
            mmproj_filename = "mmproj-model-f16.gguf"
            
            print(f"Downloading CLIP model from {mmproj_repo}...")
            # Download the CLIP model
            clip_model_path = hf_hub_download(
                repo_id=mmproj_repo,
                filename=mmproj_filename,
            )
            
            print(f"Downloaded CLIP model to: {clip_model_path}")
            
            # Initialize llama.cpp model with Gemma3ChatHandler for multimodal support
            print(f"Initializing Gemma3ChatHandler with clip model: {clip_model_path}")
            self.chat_handler = Gemma3ChatHandler(clip_model_path=clip_model_path)
            
            print("Downloading and initializing Gemma model...")
            self.model = Llama.from_pretrained(
                repo_id="ggml-org/gemma-3-27b-it-GGUF",
                filename="gemma-3-27b-it-Q4_K_M.gguf",
                verbose=True,
                n_gpu_layers=-1,
                n_ctx=32768,
                chat_handler=self.chat_handler
            )
            print("Model initialization complete!")
        except Exception as e:
            print(f"Error during setup: {e}")
            raise

    async def run(self, input_data: AppInput, metadata) -> AsyncGenerator[AppOutput, None]:
        messages = Gemma3MessageBuilder(input_data)
        for output in stream_generate(self.model, messages, AppOutput):
            yield output

    async def unload(self):
        del self.model
        del self.chat_handler
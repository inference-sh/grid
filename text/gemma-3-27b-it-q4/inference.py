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

class AppInput(LLMInputWithImage):
    # enable_thinking: bool = Field(
    #     description="Whether to enable thinking mode for complex reasoning",
    #     default=False
    # )
    pass

class AppOutput(BaseAppOutput):
    response: str
    # thinking_content: Optional[str] = None

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

    async def run(self, input: AppInput) -> AsyncGenerator[AppOutput, None]:
        # Build messages list with proper multimodal format
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": input.system_prompt}],
            }
        ]

        # Add context messages
        for msg in input.context:
            message_content = []
            if msg.text:
                message_content.append({"type": "text", "text": msg.text})
            if msg.image and msg.image.path:
                message_content.append({"type": "image", "url": msg.image.path})
            if msg.image and msg.image.uri:
                message_content.append({"type": "image", "url": msg.image.uri})
            messages.append({
                "role": msg.role,
                "content": message_content
            })

        # Add user message with text and image if provided
        user_content = []
        user_text = input.text
        
        # Add thinking instructions if enabled
        if hasattr(input, 'enable_thinking') and input.enable_thinking:
            user_text = f"{user_text} /think"
        
        if user_text:
            user_content.append({"type": "text", "text": user_text})
        if input.image and input.image.path:
            user_content.append({"type": "image", "url": input.image.path})
        if input.image and input.image.uri:
            user_content.append({"type": "image", "url": input.image.uri})
            
        messages.append({"role": "user", "content": user_content})

        print(f"Sending messages to model: {messages}")
        
        response_queue: "Queue[Optional[str]]" = Queue()

        def generation_thread():
            try:
                print("Starting generation...")
                for chunk in self.model.create_chat_completion(
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
                print("Generation completed successfully")
            except Exception as e:
                print(f"Error during generation: {e}")
                # Put the error message in the queue so the user sees it
                response_queue.put(f"\nError during generation: {str(e)}")
            finally:
                response_queue.put(None)

        thread = Thread(target=generation_thread, daemon=True)
        thread.start()

        buffer = ""
        thinking_content = ""
        in_thinking = hasattr(input, 'enable_thinking') and input.enable_thinking
        
        loop = asyncio.get_event_loop()
        while True:
            piece = await loop.run_in_executor(None, response_queue.get)
            if piece is None:
                break
                
            # Clean up any special tokens that might be in the response
            piece = piece.replace("<|im_end|>", "").replace("<|im_start|>", "").replace("<end_of_turn>", "")
            
            # Parse thinking content if enabled
            if hasattr(input, 'enable_thinking') and input.enable_thinking:
                # Check for </think> token to switch from thinking to response
                if "</think>" in piece:
                    parts = piece.split("</think>")
                    if in_thinking:
                        thinking_content += parts[0]
                        # Clean up any <think> tag from thinking content
                        thinking_content = thinking_content.replace("<think>", "")
                        buffer += parts[1] if len(parts) > 1 else ""
                        in_thinking = False
                    else:
                        buffer += piece
                else:
                    if in_thinking:
                        # Clean up any <think> tag while accumulating thinking content
                        piece = piece.replace("<think>", "")
                        thinking_content += piece
                    else:
                        buffer += piece
            else:
                buffer += piece
                
            output = {"response": buffer.strip()}
            if hasattr(input, 'enable_thinking') and input.enable_thinking and thinking_content:
                output["thinking_content"] = thinking_content.strip()
            yield AppOutput(**output)

        thread.join()

    async def unload(self):
        del self.model
        del self.chat_handler
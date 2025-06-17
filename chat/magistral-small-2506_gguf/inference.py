import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from inferencesh import BaseApp, BaseAppOutput, ContextMessage, LLMInput
from pydantic import Field, BaseModel
from typing import AsyncGenerator, List, Optional
from queue import Queue
from threading import Thread
import asyncio
import PIL
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import os.path
import base64
from llama_cpp.llama_chat_format import Jinja2ChatFormatter
from contextlib import ExitStack

def strftime_now(*args, **kwargs):
    return datetime.now().strftime(**kwargs)

configs = {
    "default": {
        "repo_id": "lmstudio-community/Magistral-Small-2506-GGUF",
        "model_filename": "Magistral-Small-2506-F16.gguf",
    },
    "q8": {
        "repo_id": "lmstudio-community/Magistral-Small-2506-GGUF",
        "model_filename": "Magistral-Small-2506-Q8_0.gguf",
    },
    "q6": {
        "repo_id": "lmstudio-community/Magistral-Small-2506-GGUF",
        "model_filename": "Magistral-Small-2506-Q6_K.gguf",
    },
    "q4": {
        "repo_id": "lmstudio-community/Magistral-Small-2506-GGUF",
        "model_filename": "Magistral-Small-2506-Q4_K_M.gguf",
    },
    "q3": {
        "repo_id": "lmstudio-community/Magistral-Small-2506-GGUF",
        "model_filename": "Magistral-Small-2506-Q3_K_L.gguf",
    },
}

MAGISTRAL_JINJA_TEMPLATE = ("{{ '<bos>' }}"
        "{%- if messages[0]['role'] == 'system' -%}"
        "{%- if messages[0]['content'] is string -%}"
        "{%- set first_user_prefix = messages[0]['content'] + '\n\n' -%}"
        "{%- else -%}"
        "{%- set first_user_prefix = messages[0]['content'][0]['text'] + '\n\n' -%}"
        "{%- endif -%}"
        "{%- set loop_messages = messages[1:] -%}"
        "{%- else -%}"
        "{%- set first_user_prefix = \"\" -%}"
        "{%- set loop_messages = messages -%}"
        "{%- endif -%}"
        "{%- for message in loop_messages -%}"
        "{%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}"
        "{{ raise_exception(\"Conversation roles must alternate user/assistant/user/assistant/...\") }}"
        "{%- endif -%}"
        "{%- if (message['role'] == 'assistant') -%}"
        "{%- set role = \"model\" -%}"
        "{%- else -%}"
        "{%- set role = message['role'] -%}"
        "{%- endif -%}"
        "{{ '<start_of_turn>' + role + '\n' + (first_user_prefix if loop.first else \"\") }}"
        "{%- if message['content'] is string -%}"
        "{{ message['content'] | trim }}"
        "{%- elif message['content'] is iterable -%}"
        "{%- for item in message['content'] -%}"
        "{%- if item['type'] == 'image_url' -%}"
        "{{ '<start_of_image>' }}"
        "{%- elif item['type'] == 'text' -%}"
        "{{ item['text'] | trim }}"
        "{%- endif -%}"
        "{%- endfor -%}"
        "{%- else -%}"
        "{{ raise_exception(\"Invalid content type\") }}"
        "{%- endif -%}"
        "{{ '<end_of_turn>\n' }}"
        "{%- endfor -%}"
        "{%- if add_generation_prompt -%}"
        "{{ '<start_of_turn>model\n' }}"
        "{%- endif -%}")


jinja_formatter = Jinja2ChatFormatter(
    MAGISTRAL_JINJA_TEMPLATE,
    eos_token="<end_of_turn>",
    bos_token="<bos>"
)

class AppInput(LLMInput):
    # enable_thinking: bool = Field(
    #     description="Whether to enable thinking mode for complex reasoning",
    #     default=False
    # )
    pass

class AppOutput(BaseAppOutput):
    response: str
    thinking_content: Optional[str] = None
    
def MessageBuilder(input_data: AppInput):
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": """A user will ask you to solve a task. You should first draft your thinking process (inner monologue) until you have derived the final answer. Afterwards, write a self-contained summary of your thoughts (i.e. your summary should be succinct but contain all the critical steps you needed to reach the conclusion). You should use Markdown and Latex to format your response. Write both your thoughts and summary in the same language as the task posed by the user.

Your thinking process must follow the template below:
<think>
Your thoughts or/and draft, like working through an exercise on scratch paper. Be as casual and as long as you want until you are confident to generate a correct answer.
</think>

Here, provide a concise summary that reflects your reasoning and presents a clear final answer to the user.

Problem:"""}],
        }
    ]

    # Add context messages
    for msg in input_data.context:
        message_content = []
        if msg.text:
            message_content.append({"type": "text", "text": msg.text})
        messages.append({
            "role": msg.role,
            "content": message_content
        })

    # Add user message with text and image if provided
    user_content = []
    user_text = input_data.text
    if user_text:
        user_content.append({"type": "text", "text": user_text})
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
                max_tokens=40960,
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

    thinking_content = ""
    response = ""
    total_piece = ""
    in_thinking = False

    try:
        while True:
            piece = response_queue.get()
            print(f"PIECE: {piece}")
            if piece is None:
                break
            if thread_exception:
                raise thread_exception
            # Clean up any special tokens

            total_piece += piece
            
            piece = piece.replace("<|im_end|>", "").replace("<|im_start|>", "").replace("<end_of_turn>", "").replace("<start_of_turn>", "")

            if "<think>" not in total_piece:
                # Wait for <think> to be fully shown
                continue
            else:
                
                if "</think>" not in total_piece:
                    # Wait for </think> to be fully shown
                    thinking_content += piece
                    in_thinking = True
                else:
                    in_thinking = False
                    response += piece

                thinking_content.replace("<think>", "").replace("</think>", "").replace("</think", "")
                response = response.replace("<start_of_turn>", "").replace("<end_of_turn>", "").replace("</start_of_turn>", "").replace("</end_of_turn>", "")

                yield AppOutput(
                    response=response,
                    thinking_content=thinking_content.strip() if thinking_content else None
                )

                if "</start_of_turn>" in total_piece:
                    break


                
                
                
        if thread_exception:
            raise thread_exception
    finally:
        thread.join(timeout=10.0)

class App(BaseApp):
    async def setup(self, metadata):
        self.variant_config = configs[metadata.app_variant]
    
        try:            
            print("Downloading and initializing Magistral model...")
            self.model = Llama.from_pretrained(
                repo_id=self.variant_config["repo_id"],
                filename=self.variant_config["model_filename"],
                verbose=False,
                n_gpu_layers=-1,
                n_ctx=49152,
                chat_handler=jinja_formatter.to_chat_handler()
            )
            print("Model initialization complete!")
        except Exception as e:
            print(f"Error during setup: {e}")
            raise

    async def run(self, input_data: AppInput, metadata) -> AsyncGenerator[AppOutput, None]:
        messages = MessageBuilder(input_data)
        for output in stream_generate(self.model, messages, AppOutput):
            yield output

    async def unload(self):
        del self.model
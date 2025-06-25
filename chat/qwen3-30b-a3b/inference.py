import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from inferencesh import BaseApp, LLMInput, LLMOutput
from inferencesh.models.llm import build_messages, stream_generate
from typing import AsyncGenerator
from llama_cpp import Llama

import os.path

configs = {
    "default": {
        "repo_id": "lmstudio-community/Qwen3-30B-A3B-GGUF",
        "model_filename": "Qwen3-30B-A3B-Q8_0.gguf",
    },
    "q6_k": {
        "repo_id": "lmstudio-community/Qwen3-30B-A3B-GGUF",
        "model_filename": "Qwen3-30B-A3B-Q6_K.gguf",
    },
    "q4_k_m": {
        "repo_id": "lmstudio-community/Qwen3-30B-A3B-GGUF",
        "model_filename": "Qwen3-30B-A3B-Q4_K_M.gguf",
    },
    "q3_k_l": {
        "repo_id": "lmstudio-community/Qwen3-30B-A3B-GGUF",
        "model_filename": "Qwen3-30B-A3B-Q3_K_L.gguf",
    }
}

class AppInput(LLMInput):
    pass

class AppOutput(LLMOutput):
    pass

def transform_response(piece: str, buffer: str) -> tuple[str, LLMOutput]:
    """Transform each response piece and return updated buffer and output."""
    cleaned = (piece.replace("<|im_end|>", "")
                  .replace("<|im_start|>", "")
                  .replace("<end_of_turn>", "")
                  .replace("<eos>", ""))
    new_buffer = buffer + cleaned
    return new_buffer, LLMOutput(
        response=new_buffer,
        thinking_content="",
    )

def log_layers(model: Llama):
    total_layers = model.n_layer
    print(f"Total layers: {total_layers}")
    device_layers = {
        0: 0,  # CPU
        1: 0,  # GPU
        2: 0,  # ACCEL
    }
    for i in range(total_layers):
        dev_layer = model.dev_layer(i)
        # Use the enum value (integer) as the key
        dev_layer_value = int(dev_layer.value)
        device_layers[dev_layer_value] += 1
    print(f"Layers on CPU: {device_layers[0]}, GPU: {device_layers[1]}, ACCEL: {device_layers[2]}")

class App(BaseApp):
    def __init__(self):
        super().__init__()
        self.last_context_size = None

    async def setup(self, metadata, context_size=None):
        config = configs[metadata.variant]
        try:            
            # Use context_size from input if provided, else default
            n_ctx = context_size if context_size is not None else 32768
            self.last_context_size = n_ctx

            print("Downloading and initializing Gemma model...")
            self.model = Llama.from_pretrained(
                repo_id=config['repo_id'],
                filename=config['model_filename'],
                verbose=False,
                n_gpu_layers=-1,
                n_ctx=n_ctx,
            )
            print("Model initialization complete!")
            log_layers(self.model)
        except Exception as e:
            print(f"Error during setup: {e}")
            raise

    async def run(self, input_data: AppInput, metadata) -> AsyncGenerator[AppOutput, None]:
        # If context_size changed, re-setup the model
        if not hasattr(self, 'last_context_size') or input_data.context_size != self.last_context_size:
            print(f"Context size changed (was {getattr(self, 'last_context_size', None)}, now {input_data.context_size}), triggering re-setup.")
            self.model.recreate_context(
                n_ctx=input_data.context_size,
            )

        # Build messages using SDK helper
        messages = build_messages(input_data)

        # Stream generate with user-specified parameters using SDK helper
        generator = stream_generate(
            model=self.model,
            messages=messages,
            output_cls=AppOutput,
            temperature=input_data.temperature,
            top_p=input_data.top_p,
            max_tokens=input_data.max_tokens,
            stop=['<end_of_turn>', '<eos>'],
            transform_response=transform_response
        )
        
        try:
            async for output in generator:
                yield output
        except Exception as e:
            print(f"[ERROR] Exception caught in run method: {type(e).__name__}: {str(e)}")
            raise

    async def unload(self):
        del self.model
        del self.chat_handler
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from inferencesh import BaseApp
from inferencesh.models.llm import (
    LLMInput,
    LLMOutput,
    ImageCapabilityMixin,
    ReasoningCapabilityMixin,
    ToolsCapabilityMixin,
    build_messages,
    stream_generate,
    ResponseTransformer
)
from typing import AsyncGenerator
from pydantic import Field

from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import os.path

configs = {
    "default": {
        "repo_id": "microsoft/phi-4-gguf",
        "model_filename": "phi-4-bf16.gguf",
    },
    "q8_0": {
        "repo_id": "microsoft/phi-4-gguf",
        "model_filename": "phi-4-Q8_0.gguf",
    },
    "q6_k": {
        "repo_id": "microsoft/phi-4-gguf",
        "model_filename": "phi-4-Q6_K.gguf",
    },
    "q4_k": {
        "repo_id": "microsoft/phi-4-gguf",
        "model_filename": "phi-4-Q4_K.gguf",
    },
    "q3_k_l": {
        "repo_id": "microsoft/phi-4-gguf",
        "model_filename": "phi-4-Q3_K_L.gguf",
    },
}

class AppInput(LLMInput, ImageCapabilityMixin, ReasoningCapabilityMixin, ToolsCapabilityMixin):
    """Phi-4 input model with image, reasoning and tools support."""
    system_prompt: str = Field(
        description="The system prompt to use for the model",
        default="You are Phi-4, a helpful and knowledgeable AI assistant.",
    )
    pass

class AppOutput(LLMOutput):
    """Phi-4 output model with token usage and timing information."""
    pass

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

    async def setup(self, metadata):
        self.variant_config = configs[metadata.app_variant]
        # Use context_size from input if provided, else default
        n_ctx = 4096
        self.last_context_size = n_ctx
        try:
            # Check if model file is available locally
            try:
                local_path = hf_hub_download(
                    repo_id=self.variant_config["repo_id"],
                    filename=self.variant_config["model_filename"],
                    local_files_only=True
                )
                print(f"Model is already available locally at: {local_path}")
                model_is_available = True
            except Exception:
                print("Model file not found locally, will be downloaded by Llama.from_pretrained.")
                model_is_available = False

            if model_is_available:
                print("Loading previously downloaded model from cache...")
            else:
                print("Downloading and initializing Phi-4 model...")

            self.model = Llama.from_pretrained(
                repo_id=self.variant_config["repo_id"],
                filename=self.variant_config["model_filename"],
                verbose=False,
                n_gpu_layers=-1,
                n_ctx=n_ctx,
                local_files_only=model_is_available,
            )
            print("Model initialization complete!")
            log_layers(self.model)
        except Exception as e:
            print(f"Error during setup: {e}")
            raise

    async def run(self, input_data: AppInput, metadata) -> AsyncGenerator[LLMOutput, None]:
        # If context_size changed, re-setup the model
        if not hasattr(self, 'last_context_size') or input_data.context_size != self.last_context_size:
            print(f"Context size changed (was {getattr(self, 'last_context_size', None)}, now {input_data.context_size}), triggering re-setup.")
            self.model.recreate_context(
                n_ctx=input_data.context_size,
            )     

        # Build messages using SDK helper
        messages = build_messages(input_data)

        # Create transformer instance with our output class
        transformer = ResponseTransformer(output_cls=AppOutput)

        # Stream generate with user-specified parameters using SDK helper
        generator = stream_generate(
            model=self.model,
            messages=messages,
            transformer=transformer,
            temperature=input_data.temperature,
            top_p=input_data.top_p,
            stop=['<|im_end|>', '<end_of_turn>'],
            output_cls=AppOutput
        )
        
        try:
            for output in generator:
                yield output
        except Exception as e:
            print(f"[ERROR] Exception caught in run method: {type(e).__name__}: {str(e)}")
            raise  # Re-raise the exception to propagate it upstream

    async def unload(self):
        del self.model
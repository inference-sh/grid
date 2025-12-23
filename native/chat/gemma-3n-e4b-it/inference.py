import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from inferencesh import BaseApp
from inferencesh.models.llm import (
    LLMInput,
    LLMOutput,
    ImageCapabilityMixin,
    build_messages,
    stream_generate,
    ResponseTransformer
)
from typing import AsyncGenerator
from pydantic import Field

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Jinja2ChatFormatter
# from llama_cpp.llama_chat_format import Gemma3ChatHandler
# from huggingface_hub import hf_hub_download
import os.path

# Configuration for the model
# vision_config = {
#     "mmproj_repo": "ggml-org/gemma-3-27b-it-GGUF",
#     "mmproj_filename": "mmproj-model-f16.gguf"
# }


configs = {
    "default": {
        "repo_id": "unsloth/gemma-3n-E4B-it-GGUF",
        "model_filename": "gemma-3n-E4B-it-F16.gguf",
    },
    "q8_0": {
        "repo_id": "unsloth/gemma-3n-E4B-it-GGUF",
        "model_filename": "gemma-3n-E4B-it-Q8_0.gguf",
    },
    "q6_k": {
        "repo_id": "unsloth/gemma-3n-E4B-it-GGUF",
        "model_filename": "gemma-3n-E4B-it-Q6_K.gguf",
    },
    "q4_k_m": {
        "repo_id": "unsloth/gemma-3n-E4B-it-GGUF",
        "model_filename": "gemma-3n-E4B-it-Q4_K_M.gguf",
    }
}

class AppInput(LLMInput, ImageCapabilityMixin):
    """Gemma 3N E4B IT input model with image and tools support."""
    system_prompt: str = Field(
        description="The system prompt to use for the model",
        default="You are Gemma, a helpful and knowledgeable AI assistant.",
    )
    pass

class AppOutput(LLMOutput):
    """Gemma 3N E4B IT output model with token usage and timing information."""
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


# Load the template and system prompt
with open(os.path.join(os.path.dirname(__file__), "templates/template.jinja"), "r") as f:
    GEMMA_JINJA_TEMPLATE = f.read()

jinja_formatter = Jinja2ChatFormatter(
    GEMMA_JINJA_TEMPLATE,
    eos_token="<end_of_turn>",
    bos_token="<bos>"
)

class App(BaseApp):
    def __init__(self):
        super().__init__()
        self.last_context_size = None

    async def setup(self, metadata):
        self.variant_config = configs[metadata.app_variant]
        # Use context_size from input if provided, else default
        n_ctx = 4096
        # We don't have the projection model for this model yet
        # Download CLIP model from Hugging Face Hub
        # print(f"Downloading CLIP model from {vision_config['mmproj_repo']}...")
        # clip_model_path = hf_hub_download(
        #     repo_id=vision_config['mmproj_repo'],
        #     filename=vision_config['mmproj_filename'],
        # )
        # print(f"Downloaded CLIP model to: {clip_model_path}")
        
        # Initialize Gemma3ChatHandler for multimodal support
        # print(f"Initializing Gemma3ChatHandler with clip model: {clip_model_path}")
        # self.chat_handler = Gemma3ChatHandler(clip_model_path=clip_model_path)
        
        # Use context_size from input if provided, else default
        self.last_context_size = n_ctx

        print("Downloading and initializing Gemma model...")
        self.model = Llama.from_pretrained(
            repo_id=self.variant_config['repo_id'],
            filename=self.variant_config['model_filename'],
            n_gpu_layers=-1,
            n_ctx=n_ctx,
            verbose=True,
            chat_handler=jinja_formatter.to_chat_handler()
            # chat_handler=self.chat_handler
        )
        print("Model initialization complete!")
        log_layers(self.model)


    async def run(self, input_data: AppInput, metadata) -> AsyncGenerator[AppOutput, None]:
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
            stop=['<end_of_turn>', '<eos>'],
            output_cls=AppOutput,
        )
        
        try:
            for output in generator:
                yield output
        except Exception as e:
            print(f"[ERROR] Exception caught in run method: {type(e).__name__}: {str(e)}")
            raise

    async def unload(self):
        del self.model
        del self.chat_handler
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from inferencesh import BaseApp
from inferencesh.models.llm import (
    LLMInput,
    LLMOutput,
    ReasoningCapabilityMixin,
    ReasoningMixin,
    ToolsCapabilityMixin,
    ToolCallsMixin,
    build_messages,
    stream_generate,
    ResponseTransformer
)
from typing import AsyncGenerator
from pydantic import Field

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Jinja2ChatFormatter

import os.path

configs = {
    "default": {
        "repo_id": "unsloth/GLM-4.5-Air-GGUF",
        "model_filename": "Q4_0/GLM-4.5-Air-Q4_0-00001-of-00002.gguf",
        "additional_files": ["Q4_0/GLM-4.5-Air-Q4_0-00002-of-00002.gguf"],
    },
    "q3_k_m": {
        "repo_id": "unsloth/GLM-4.5-Air-GGUF",
        "model_filename": "Q3_K_M/GLM-4.5-Air-Q3_K_M-00001-of-00002.gguf",
        "additional_files": ["Q3_K_M/GLM-4.5-Air-Q3_K_M-00002-of-00002.gguf"],
    },
    "q3_k_s": {
        "repo_id": "unsloth/GLM-4.5-Air-GGUF",
        "model_filename": "Q3_K_S/GLM-4.5-Air-Q3_K_S-00001-of-00002.gguf",
        "additional_files": ["Q3_K_S/GLM-4.5-Air-Q3_K_S-00002-of-00002.gguf"],
    },
    "q4_0": {
        "repo_id": "unsloth/GLM-4.5-Air-GGUF",
        "model_filename": "Q4_0/GLM-4.5-Air-Q4_0-00001-of-00002.gguf",
        "additional_files": ["Q4_0/GLM-4.5-Air-Q4_0-00002-of-00002.gguf"],
    },
    "q4_1": {
        "repo_id": "unsloth/GLM-4.5-Air-GGUF",
        "model_filename": "Q4_1/GLM-4.5-Air-Q4_1-00001-of-00002.gguf",
        "additional_files": ["Q4_1/GLM-4.5-Air-Q4_1-00002-of-00002.gguf"],
    },
    "q4_k_m": {
        "repo_id": "unsloth/GLM-4.5-Air-GGUF",
        "model_filename": "Q4_K_M/GLM-4.5-Air-Q4_K_M-00001-of-00002.gguf",
        "additional_files": ["Q4_K_M/GLM-4.5-Air-Q4_K_M-00002-of-00002.gguf"],
    },
    "q4_k_s": {
        "repo_id": "unsloth/GLM-4.5-Air-GGUF",
        "model_filename": "Q4_K_S/GLM-4.5-Air-Q4_K_S-00001-of-00002.gguf",
        "additional_files": ["Q4_K_S/GLM-4.5-Air-Q4_K_S-00002-of-00002.gguf"],
    }
}


# Load the template and system prompt
with open(os.path.join(os.path.dirname(__file__), "templates/template.jinja"), "r") as f:
    MAGISTRAL_JINJA_TEMPLATE = f.read()

with open(os.path.join(os.path.dirname(__file__), "templates/system_prompt.txt"), "r") as f:
    SYSTEM_PROMPT = f.read()

jinja_formatter = Jinja2ChatFormatter(
    MAGISTRAL_JINJA_TEMPLATE,
    eos_token="<end_of_turn>",
    bos_token="<bos>"
)

class AppInput(LLMInput, ReasoningCapabilityMixin, ToolsCapabilityMixin):
    """GLM-4.5-Air input model with image and tools support."""
    system_prompt: str = Field(
        description="The system prompt to use for the model",
        default=SYSTEM_PROMPT,
    )
    pass

class AppOutput(ReasoningMixin, ToolCallsMixin, LLMOutput):
    """GLM-4.5-Air output model with token usage and timing information."""
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

        print("Downloading and initializing GLM-4.5-Air model...")
        # Add our template to model metadata
        self.model = Llama.from_pretrained(
            repo_id=self.variant_config['repo_id'],
            filename=self.variant_config['model_filename'],
            additional_files=self.variant_config['additional_files'],
            verbose=True,
            n_gpu_layers=-1, # -1 means use all GPU layers available
            n_ctx=n_ctx,
            # chat_format="gguf-function-calling",
            # chat_format="chatml-function-calling",
            # metadata={"tokenizer.chat_template": MAGISTRAL_JINJA_TEMPLATE},
            chat_handler=jinja_formatter.to_chat_handler()
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
            self.last_context_size = input_data.context_size

        # Build messages using SDK helper with /nothink appended when reasoning is disabled
        def transform_message(text: str) -> str:
            if not input_data.reasoning and not text.endswith("/nothink"):
                return f"{text}/nothink"
            return text
            
        messages = build_messages(input_data, transform_user_message=transform_message)

        # Create transformer instance with our output class
        transformer = ResponseTransformer(output_cls=AppOutput)

        # Stream generate with user-specified parameters using SDK helper
        generator = stream_generate(
            model=self.model,
            messages=messages,
            transformer=transformer,
            temperature=input_data.temperature,
            top_p=input_data.top_p,
            tools=input_data.tools,
            tool_choice="auto",
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
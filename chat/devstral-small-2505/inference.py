import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from inferencesh import BaseApp, ContextMessage, LLMInput, LLMOutput
from inferencesh.models.llm import build_messages, stream_generate, ResponseTransformer
from pydantic import Field
from typing import AsyncGenerator

from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import os.path
from llama_cpp.llama_chat_format import Jinja2ChatFormatter

configs = {
    "default": {
        "repo_id": "mistralai/Devstral-Small-2505_gguf",
        "model_filename": "devstral.gguf",
    },
    "q8": {
        "repo_id": "mistralai/Devstral-Small-2505_gguf",
        "model_filename": "devstralQ8_0.gguf",
    },
    "q5": {
        "repo_id": "mistralai/Devstral-Small-2505_gguf",
        "model_filename": "devstralQ5_K_M.gguf",
    },
    "q4": {
        "repo_id": "mistralai/Devstral-Small-2505_gguf",
        "model_filename": "devstralQ4_K_M.gguf",
    },
    "q4_0": {
        "repo_id": "mistralai/Devstral-Small-2505_gguf",
        "model_filename": "devstralQ4_0.gguf",
    },
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

class AppInput(LLMInput):
    system_prompt: str = Field(
        description="The system prompt to use for the model",
        default=SYSTEM_PROMPT,
        examples=[]
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
    temperature: float = Field(
        description="The temperature to use for the model",
        default=0.7
    )
    top_p: float = Field(
        description="The top-p to use for the model",
        default=0.95
    )
    max_tokens: int = Field(
        description="The maximum number of tokens to generate",
        default=40960
    )
    context_size: int = Field(
        description="The maximum number of tokens to use for the context (changing this will cause a model re-setup)",
        min_value=4096,
        max_value=49152,
        default=4096,
    )

class AppOutput(LLMOutput):
    pass

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
                print("Downloading and initializing Devstral model...")

            self.model = Llama.from_pretrained(
                repo_id=self.variant_config["repo_id"],
                filename=self.variant_config["model_filename"],
                verbose=False,
                n_gpu_layers=-1,
                n_ctx=n_ctx,
                local_files_only=model_is_available,
                chat_handler=jinja_formatter.to_chat_handler()
            )
            print("Model initialization complete!")
            total_layers = self.model.n_layer
            print(f"Total layers: {total_layers}")
            device_layers = {
                0: 0,  # CPU
                1: 0,  # GPU
                2: 0,  # ACCEL
            }
            for i in range(total_layers):
                dev_layer = self.model.dev_layer(i)
                # Use the enum value (integer) as the key
                dev_layer_value = int(dev_layer.value)
                device_layers[dev_layer_value] += 1
            print(f"Layers on CPU: {device_layers[0]}, GPU: {device_layers[1]}, ACCEL: {device_layers[2]}")
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
            max_tokens=input_data.max_tokens,
            stop=['<end_of_turn>']
        )
        
        try:
            for output in generator:
                yield output
        except Exception as e:
            print(f"[ERROR] Exception caught in run method: {type(e).__name__}: {str(e)}")
            raise  # Re-raise the exception to propagate it upstream

    async def unload(self):
        del self.model
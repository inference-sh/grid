import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput
from pydantic import Field, BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from typing import AsyncGenerator
from threading import Thread

class ContextMessage(BaseModel):
    role: str = Field(
        description="The role of the message",
        enum=["user", "assistant", "system"]
    )
    content: str = Field(
        description="The content of the message"
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
                {"role": "user", "content": "What is the capital of France?"}, 
                {"role": "assistant", "content": "The capital of France is Paris."}
            ],
            [
                {"role": "user", "content": "What is the weather like today?"}, 
                {"role": "assistant", "content": "I apologize, but I don't have access to real-time weather information. You would need to check a weather service or app to get current weather conditions for your location."}
            ],
            [
                {"role": "user", "content": "Can you help me write a poem about spring?"}, 
                {"role": "assistant", "content": "Here's a short poem about spring:\n\nGreen buds awakening,\nSoft rain gently falling down,\nNew life springs anew.\n\nWarm sun breaks through clouds,\nBirds return with joyful song,\nNature's sweet rebirth."}
            ],
            [
                {"role": "user", "content": "Explain quantum computing in simple terms"}, 
                {"role": "assistant", "content": "Quantum computing is like having a super-powerful calculator that can solve many problems at once instead of one at a time. While regular computers use bits (0s and 1s), quantum computers use quantum bits or \"qubits\" that can be both 0 and 1 at the same time - kind of like being in two places at once! This allows them to process huge amounts of information much faster than regular computers for certain types of problems."}
            ]
        ],
        default=[]
    )
    user_prompt: str = Field(
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

class App(BaseApp):
    async def setup(self):
        """Initialize your model and resources here."""
        self.device = "cuda" # the device to load the model onto

        self.model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-3-27b-it-qat-q4_0-gguf",
            device_map="auto",
            torch_dtype="auto"
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-27b-it-qat-q4_0-gguf")

    async def run(self, input_data: AppInput) -> AsyncGenerator[AppOutput, None]:
        """Run prediction on the input data."""
        messages = [
            {"role": "system", "content": input_data.system_prompt},
            *input_data.context,
            {"role": "user", "content": input_data.user_prompt}
        ]

        # Convert messages to Gemma's chat format
        chat = []
        for msg in messages:
            if msg["role"] == "system":
                chat.append({"role": "user", "content": msg["content"]})
                chat.append({"role": "assistant", "content": "Understood."})
            else:
                chat.append(msg)

        # Apply chat template and prepare inputs
        text = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, decode_kwargs={"skip_special_tokens": True})
        
        generation_kwargs = dict(
            input_ids=model_inputs.input_ids,
            streamer=streamer,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.95
        )

        # Run generation in a separate thread to allow non-blocking streaming
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # Collect the generated text
        response = ""
        for new_text in streamer:
            # Remove any Gemma-specific special tokens
            new_text = new_text.replace("<|im_end|>", "").replace("<|im_start|>", "").replace("<end_of_turn>", "")
            response += new_text
            yield AppOutput(response=response)

    async def unload(self):
        """Clean up resources here."""
        pass
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field, BaseModel
from transformers import AutoModelForCausalLM, AutoProcessor, TextIteratorStreamer
from typing import AsyncGenerator, Optional
from threading import Thread
import PIL
# class ContextMessageContent(BaseModel):
#     type: str = Field(
#         description="The type of the message",
#         enum=["text", "image", "video", "audio"]
#     )
#     text: str = Field(
#         description="The text content of the message"
#     )
#     url: str] = Field(
#         description="The image content of the message"
#     )

class ContextMessage(BaseModel):
    role: str = Field(
        description="The role of the message",
        enum=["user", "assistant", "system"]
    )
    text: str = Field(
        description="The text content of the message"
    )
    image: Optional[File] = Field(
        description="The image url of the message",
        default=None
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
    image: Optional[File] = Field(
        description="The image to use for the model",
        default=None
    )

class AppOutput(BaseAppOutput):
    response: str

class App(BaseApp):
    async def setup(self):
        """Initialize your model and resources here."""
        self.device = "cuda" # the device to load the model onto

        model_name = "google/gemma-3-27b-it"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto"
        ).to(self.device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_name)

    async def run(self, input_data: AppInput) -> AsyncGenerator[AppOutput, None]:
        """Run prediction on the input data."""
  
        # Build messages list
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
            if msg.image and msg.image.uri:
                message_content.append({"type": "image", "url": msg.image})
            messages.append({
                "role": msg.role,
                "content": message_content
            })

        # Add user message with text and media if provided
        user_content = []
        if input_data.text:
            user_content.append({"type": "text", "text": input_data.text})
        if input_data.image and input_data.image.uri:
            user_content.append({"type": "image", "url": input_data.image.uri})
        messages.append({"role": "user", "content": user_content})

        print(messages)

        # Apply chat template and prepare inputs
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        print(prompt)
        images = [PIL.Image.open(input_data.image.path).convert("RGB")] if input_data.image else None
        model_inputs = self.processor(
            text=prompt,
            images=images,
            return_tensors="pt"
        ).to(self.device)
        streamer = TextIteratorStreamer(self.processor.tokenizer, skip_prompt=True, decode_kwargs={"skip_special_tokens": True})
        
        generation_kwargs = dict(
            **model_inputs,
            streamer=streamer,
            max_new_tokens=20736,
            do_sample=False,
            # temperature=0.7,
            # top_p=0.95
        )

        # Run generation in a separate thread to allow non-blocking streaming
        thread_exception = None
        
        def run_generation():
            nonlocal thread_exception
            try:
                self.model.generate(**generation_kwargs)
            except Exception as e:
                thread_exception = e
                streamer.end()
        
        thread = Thread(target=run_generation)
        thread.start()

        # Collect the generated text
        response = ""
        try:
            for new_text in streamer:
                if thread_exception:
                    raise thread_exception
                
                new_text = new_text.replace("<|im_end|>", "").replace("<|im_start|>", "").replace("<end_of_turn>", "")
                response += new_text
                yield AppOutput(response=response)
            
            # Check one final time for exceptions after generation is complete
            if thread_exception:
                raise thread_exception
                
        finally:
            thread.join(timeout=2.0)  # Brief timeout just to prevent infinite hangs

    async def unload(self):
        """Clean up resources here."""
        pass
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from inferencesh import BaseApp, LLMInput, BaseAppOutput
from pydantic import Field, BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from typing import AsyncGenerator
from threading import Thread

    
class AppInput(LLMInput):
    pass
class AppOutput(BaseAppOutput):
    response: str

class App(BaseApp):
    async def setup(self, metadata):
        """Initialize your model and resources here."""
        self.device = "cuda" # the device to load the model onto

        self.model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B-Instruct", device_map="auto").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")

    async def run(self, input: AppInput) -> AsyncGenerator[AppOutput, None]:
        """Run prediction on the input data."""
        # Build messages list
        messages = [
            {
                "role": "system",
                "content": input.system_prompt
            }
        ]

        # Add context messages
        for msg in input.context:
            message = {"role": msg.role, "content": msg.text}
            messages.append(message)

        # Add user message with text and media if provided
        user_message = input.text            
        messages.append({"role": "user", "content": user_message})

        # Apply chat template and prepare inputs
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, decode_kwargs={"skip_special_tokens": True})
        generation_kwargs = dict(
            input_ids=model_inputs.input_ids,
            streamer=streamer,
            max_new_tokens=512,
            do_sample=True
        )

        # Run generation in a separate thread to allow non-blocking streaming
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # Collect the generated text
        response = ""
        for new_text in streamer:
            new_text = new_text.replace("<|im_end|>", "")
            response += new_text
            yield AppOutput(response=response)

    async def unload(self):
        """Clean up resources here."""
        pass
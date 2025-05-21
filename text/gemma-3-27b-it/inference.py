import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from inferencesh import BaseApp, BaseAppOutput, LLMInput, LLMInputWithImage, ContextMessage
from pydantic import Field
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

class AppInput(LLMInputWithImage):
    pass

class AppOutput(BaseAppOutput):
    response: str

class App(BaseApp):
    async def setup(self, metadata):
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

    async def run(self, input: AppInput) -> AsyncGenerator[AppOutput, None]:
        """Run prediction on the input data."""
  
        # Build messages list
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
            if msg.image and msg.image.uri:
                message_content.append({"type": "image", "url": msg.image.uri})
            messages.append({
                "role": msg.role,
                "content": message_content
            })

        # Add user message with text and media if provided
        user_content = []
        if input.text:
            user_content.append({"type": "text", "text": input.text})
        if input.image and input.image.uri:
            user_content.append({"type": "image", "url": input.image.uri})
        messages.append({"role": "user", "content": user_content})

        print(messages)

        # Apply chat template and prepare inputs
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        print(prompt)
        images = [PIL.Image.open(input.image.path).convert("RGB")] if input.image else None
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
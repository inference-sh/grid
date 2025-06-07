import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import torch

from inferencesh import BaseApp, BaseAppOutput, LLMInputWithImage
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    TextIteratorStreamer,
)
from typing import AsyncGenerator, Generator
from threading import Thread
from PIL import Image


class AppInput(LLMInputWithImage):
    pass


class AppOutput(BaseAppOutput):
    response: str


def Gemma3MessageBuilder(input_data: AppInput):
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
            message_content.append({"type": "image", "url": msg.image.uri})
        messages.append({"role": msg.role, "content": message_content})

    # Add user message with text and media if provided
    user_content = []
    if input_data.text:
        user_content.append({"type": "text", "text": input_data.text})
    if input_data.image and input_data.image.uri:
        user_content.append({"type": "image", "url": input_data.image.uri})
    messages.append({"role": "user", "content": user_content})
    return messages


def stream_generate(model, generation_kwargs, streamer, AppOutput):
    """Handles threaded generation and yields AppOutput as text streams in."""
    thread_exception = None

    def run_generation():
        nonlocal thread_exception
        try:
            model.generate(**generation_kwargs)
        except Exception as e:
            thread_exception = e
            streamer.end()

    thread = Thread(target=run_generation)
    thread.start()

    response = ""
    try:
        for new_text in streamer:
            if thread_exception:
                raise thread_exception

            new_text = (
                new_text.replace("<|im_end|>", "")
                .replace("<|im_start|>", "")
                .replace("<end_of_turn>", "")
            )
            response += new_text
            yield AppOutput(response=response)

        if thread_exception:
            raise thread_exception

    finally:
        thread.join(timeout=2.0)


class App(BaseApp):
    async def setup(self, metadata):
        """Initialize your model and resources here."""
        self.device = "cuda"  # the device to load the model onto

        model_name = "google/gemma-3-27b-it"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
        ).to(self.device)
        self.model.eval()
        
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            padding_side="left",
            pad_to_multiple_of=8,
        )

    async def run(
        self, input_data: AppInput, metadata
    ) -> AsyncGenerator[AppOutput, None]:
        """Run prediction on the input data."""

        # Build messages list
        messages = Gemma3MessageBuilder(input_data)

        # Apply chat template and prepare inputs
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        
        images = (
            [Image.open(input_data.image.path).convert("RGB")]
            if input_data.image
            else None
        )
        
        model_inputs = self.processor(
            text=prompt, images=images, return_tensors="pt"
        ).to(self.device)
        
        streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            skip_prompt=True,
            decode_kwargs={"skip_special_tokens": True},
        )

        generation_kwargs = dict(
            **model_inputs,
            streamer=streamer,
            max_new_tokens=20736,
            do_sample=False,
            # temperature=0.7,
            # top_p=0.95
        )

        for output in stream_generate(self.model, generation_kwargs, streamer, AppOutput):
            yield output

    async def unload(self):
        """Clean up resources here."""
        pass

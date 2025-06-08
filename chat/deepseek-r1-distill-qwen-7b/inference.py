import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from inferencesh import BaseApp, BaseAppOutput, LLMInput
from pydantic import Field
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from typing import AsyncGenerator, Optional
from threading import Thread

class AppInput(LLMInput):
    enable_thinking: bool = Field(
        description="Whether to enable thinking mode for complex reasoning",
        default=True
    )

class AppOutput(BaseAppOutput):
    response: str
    thinking_content: Optional[str] = None

def build_messages(input_data: AppInput):
    messages = [
        {"role": "system", "content": input_data.system_prompt}
    ]
    # Add context messages
    for msg in input_data.context:
        content = msg.text if hasattr(msg, 'text') else msg.content
        messages.append({
            "role": msg.role,
            "content": content
        })
    
    # Add user message with thinking mode
    user_prompt = input_data.text
    if input_data.enable_thinking:
        user_prompt = f"{user_prompt} /think"
    else:
        user_prompt = f"{user_prompt} /no_think"
    messages.append({"role": "user", "content": user_prompt})
    return messages

class App(BaseApp):
    async def setup(self, metadata):
        """Initialize your model and resources here."""
        self.device = "cuda" # the device to load the model onto

        self.model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", device_map="auto").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")

    async def run(self, input_data: AppInput, metadata) -> AsyncGenerator[AppOutput, None]:
        """Run prediction on the input data."""
        messages = build_messages(input_data)
        
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, decode_kwargs={"skip_special_tokens": True})
        
        generation_kwargs = dict(
            input_ids=model_inputs.input_ids,
            streamer=streamer,
            max_new_tokens=32768,
            do_sample=True,
            temperature=0.6 if input_data.enable_thinking else 0.7,
            top_p=0.95 if input_data.enable_thinking else 0.8
        )

        # Run generation in a separate thread to allow non-blocking streaming
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # Collect the generated text
        buffer = ""
        thinking_content = ""
        in_thinking = input_data.enable_thinking
        
        for new_text in streamer:
            new_text = new_text.replace(self.tokenizer.eos_token, "")
            
            if "</think>" in new_text:
                parts = new_text.split("</think>")
                if in_thinking:
                    thinking_content += parts[0].replace("<think>", "")
                    buffer = parts[1] if len(parts) > 1 else ""
                    in_thinking = False
                else:
                    buffer += new_text
            else:
                if in_thinking:
                    thinking_content += new_text.replace("<think>", "")
                else:
                    buffer += new_text
                    
            yield AppOutput(
                response=buffer.strip(),
                thinking_content=thinking_content.strip() if thinking_content else None
            )

    async def unload(self):
        """Clean up resources here."""
        del self.model
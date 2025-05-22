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

class App(BaseApp):
    async def setup(self, metadata):
        """Initialize your model and resources here."""
        self.device = "cuda"
        model_name = "Qwen/Qwen3-30B-A3B"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto"
        ).to(self.device)
        self.model.eval()

    async def run(self, input_data: AppInput, metadata) -> AsyncGenerator[AppOutput, None]:
        """Run prediction on the input data."""
        
        # Build messages list
        messages = [
            {
                "role": "system",
                "content": input_data.system_prompt
            }
        ]

        # Add context messages
        for msg in input_data.context:
            message = {"role": msg.role, "content": msg.text}
            messages.append(message)

        # Add user message with text and media if provided
        user_message = input_data.text
        if input_data.enable_thinking:
            user_message = f"{user_message} /think"
        else:
            user_message = f"{user_message} /no_think"
            
        messages.append({"role": "user", "content": user_message})

        # Apply chat template and prepare inputs
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=input_data.enable_thinking
        )
        
        model_inputs = self.tokenizer(
            text,
            return_tensors="pt"
        ).to(self.device)

        # Generation config based on thinking mode
        generation_kwargs = {
            **model_inputs,
            "max_new_tokens": 32768,
            "do_sample": True,
            "top_k": 20,
            "min_p": 0,
            "streamer": TextIteratorStreamer(self.tokenizer, skip_prompt=True, decode_kwargs={"skip_special_tokens": True})
        }
        
        if input_data.enable_thinking:
            generation_kwargs.update({
                "temperature": 0.6,
                "top_p": 0.95,
            })
        else:
            generation_kwargs.update({
                "temperature": 0.7,
                "top_p": 0.8,
            })

        # Run generation
        thread_exception = None
        streamer = generation_kwargs["streamer"]
        
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
        thinking_content = ""
        in_thinking = input_data.enable_thinking
        
        try:
            for new_text in streamer:
                if thread_exception:
                    raise thread_exception
                
                # Clean up im_end tags
                new_text = new_text.replace("<|im_end|>", "")
                
                # Check for </think> token to switch from thinking to response
                if "</think>" in new_text:
                    parts = new_text.split("</think>")
                    if in_thinking:
                        thinking_content += parts[0]
                        # Clean up any <think> tag from thinking content
                        thinking_content = thinking_content.replace("<think>", "")
                        response = parts[1] if len(parts) > 1 else ""
                        in_thinking = False
                    else:
                        response += new_text
                else:
                    if in_thinking:
                        # Clean up any <think> tag while accumulating thinking content
                        new_text = new_text.replace("<think>", "")
                        thinking_content += new_text
                    else:
                        response += new_text
                
                yield AppOutput(
                    response=response.strip(),
                    thinking_content=thinking_content.strip() if thinking_content else None
                )
            
            # Check one final time for exceptions after generation is complete
            if thread_exception:
                raise thread_exception
                
        finally:
            thread.join(timeout=2.0)

    async def unload(self):
        """Clean up resources here."""
        pass
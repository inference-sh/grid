from inferencesh import BaseApp, BaseAppInput, BaseAppOutput
from pydantic import Field
from typing import List, Any

class AppInput(BaseAppInput):
    list_input: List[Any] = Field(description="The input list to get an element from")
    index: int = Field(description="The index to get the element from (can be negative)")

class AppOutput(BaseAppOutput):
    element: Any = Field(description="The element at the specified index in the list")

# For LLM apps, you can use the LLMInput and LLMInputWithImage classes for convenience
# from inferencesh import LLMInput, LLMInputWithImage
# The LLMInput class provides a standard structure for LLM-based applications with:
# - system_prompt: Sets the AI assistant's role and behavior
# - context: List of previous conversation messages between user and assistant
# - text: The current user's input prompt
#
# Example usage:
# class AppInput(LLMInput):
#     additional_field: str = Field(description="Any additional input needed")

# The LLMInputWithImage class extends LLMInput to support image inputs by adding:
# - image: Optional File field for providing images to vision-capable models
#
# Example usage:
# class AppInput(LLMInputWithImage):
#     additional_field: str = Field(description="Any additional input needed")

# Each ContextMessage in the context list contains:
# - role: Either "user", "assistant", or "system"
# - text: The message content
#
# ContextMessageWithImage adds:
# - image: Optional File field for messages containing images



class App(BaseApp):
    async def setup(self, metadata):
        """Initialize your model and resources here."""
        # Example: Load ML model (run once)
        # self.model = SomeModel()

        # Example: Initialize tokenizer (run once)
        # self.tokenizer = Tokenizer()

        # Example: Setup cache (run once)
        # self.cache = {}

        pass

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Get the element at the specified index in the list."""
        try:
            element = input_data.list_input[input_data.index]
            return AppOutput(element=element)
        except IndexError:
            raise RuntimeError(f"Index {input_data.index} is out of range for list of length {len(input_data.list_input)}")

    async def unload(self):
        """Clean up resources here."""
        pass
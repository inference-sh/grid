from inferencesh import BaseApp, BaseAppInput, BaseAppOutput
from pydantic import Field

class AppInput(BaseAppInput):
    template: str = Field(description="The text template with placeholders like {1}, {2}, etc.")
    strings: list[str] = Field(description="List of strings to substitute into the template")

class AppOutput(BaseAppOutput):
    result: str = Field(description="The processed template with strings substituted")

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
        pass

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Process the template with the provided strings."""
        # Create a dictionary of replacements
        replacements = {str(i+1): string for i, string in enumerate(input_data.strings)}
        
        # Process the template
        result = input_data.template
        for placeholder, value in replacements.items():
            result = result.replace(f"{{{placeholder}}}", value)
        
        return AppOutput(result=result)

    async def unload(self):
        """Clean up resources here."""
        pass
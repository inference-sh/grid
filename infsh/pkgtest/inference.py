from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from typing import List, Optional

class AppInput(BaseAppInput):
    string_input: str = Field(description="The input string to process")
    number_input: int = Field(description="The input number to process")
    boolean_input: bool = Field(description="The input boolean to process")
    list_input: List[int] = Field(description="The input list to process")
    optional_input: Optional[str] = Field(None, description="The optional input to process")
    file_input: File = Field(description="The input file to process")

class AppOutput(BaseAppOutput):
    file_output: File = Field(description="an output file")
    string_output: str = Field(description="an output string")
    number_output: int = Field(description="an output number")
    boolean_output: bool = Field(description="an output boolean")
    list_output: List[int] = Field(description="an output list")
    optional_output: Optional[str] = Field(None, description="an optional output")

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
        
        # Example: Use the variant to setup the model
        # self.variant = metadata.app_variant
        # if self.variant == "default":
        #     self.config = {
        #         "model": "default-model",
        #     }
        # elif self.variant == "small":
        #     self.config = {
        #         "model": "small-model",
        #     }
        # elif self.variant == "large":
        #     self.config = {
        #         "model": "large-model",
        #     }
        pass

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Run prediction on the input data."""
        # Example: Use resources initialized in setup()
        # tokens = self.tokenizer.tokenize(input_data.string_input)
        # prediction = self.model.predict(tokens)
        # Check if input file exists and is accessible
        if not input_data.file_input.exists():
            raise RuntimeError(f"Input file does not exist at path: {input_data.file_input.path}")
        
        # File metadata is populated automatically after file is downloaded/resolved
        # Available properties:
        # - input_data.file_input.uri: Original location (URL or file path)
        # - input_data.file_input.path: Resolved local file path
        # - input_data.file_input.content_type: MIME type of the file
        # - input_data.file_input.size: File size in bytes
        # - input_data.file_input.filename: Original filename if available

        # Process string input
        # Example: Convert to uppercase
        processed_string = input_data.string_input.upper()

        # Process number input
        # Example: Double the value
        processed_number = input_data.number_input * 2

        # Process boolean input
        # Example: Invert the value
        processed_boolean = not input_data.boolean_input

        # Process list input
        # Example: Increment each element by 1
        processed_list = [x + 1 for x in input_data.list_input]

        # Process optional input
        # Example: Convert to lowercase if present, otherwise None
        processed_optional = input_data.optional_input.lower() if input_data.optional_input else None

        # Note: These are just example transformations
        # Replace with actual processing logic as needed
        # Write results to output file
        
        output_path = "/tmp/result.txt"
        with open(output_path, "w") as f:
            f.write(f"Processed: {input_data.string_input}")
        
        # Return output file path so the engine can upload the results
        # The File object will be automatically handled by the engine
        # which will upload the file to the configured storage backend
        return AppOutput(
            file_output=File(path=output_path),
            string_output=processed_string,
            number_output=processed_number,
            boolean_output=processed_boolean,
            list_output=processed_list,
            optional_output=processed_optional
        )

    async def unload(self):
        """Clean up resources here."""
        pass
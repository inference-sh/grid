from inferencesh import BaseApp, BaseAppInput, BaseAppOutput
from pydantic import Field
from typing import List
import re

class AppInput(BaseAppInput):
    text: str = Field(description="The input text to split")
    delimiter: str = Field(description="The delimiter to split the text with")
    use_regex: bool = Field(default=False, description="Whether to treat the delimiter as a regex pattern")

class AppOutput(BaseAppOutput):
    split_text: List[str] = Field(description="The text split into parts based on the delimiter")

class App(BaseApp):
    async def setup(self, metadata):
        """Initialize your model and resources here."""
        pass

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Split the input text based on the delimiter."""
        if input_data.use_regex:
            try:
                # Use regex split
                split_result = re.split(input_data.delimiter, input_data.text)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern: {str(e)}")
        else:
            # Use normal string split
            split_result = input_data.text.split(input_data.delimiter)
        
        # Remove empty strings from the result
        split_result = [part for part in split_result if part]
        
        return AppOutput(
            split_text=split_result
        )

    async def unload(self):
        """Clean up resources here."""
        pass
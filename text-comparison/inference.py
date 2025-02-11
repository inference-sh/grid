from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from enum import Enum
from pydantic import Field
from typing import Literal

class ComparisonTask(str, Enum):
    IDENTICAL = "identical"
    CONTAINS = "contains"
    PREFIX = "prefix"
    SUFFIX = "suffix"
    CASE_INSENSITIVE = "case_insensitive"

class AppInput(BaseAppInput):
    text1: str = Field(
        description="First text to compare",
        examples=["hello world"]
    )
    text2: str = Field(
        description="Second text to compare",
        examples=["hello"]
    )
    task: Literal["identical", "contains", "prefix", "suffix", "case_insensitive"] = Field(
        description="Type of comparison to perform",
        examples=["contains"]
    )

class AppOutput(BaseAppOutput):
    result: bool = Field(
        description="Comparison result"
    )

class App(BaseApp):
    async def setup(self):
        """Initialize your model and resources here."""
        pass

    async def run(self, input_data: AppInput) -> AppOutput:
        """Run prediction on the input data."""
        text1, text2 = input_data.text1, input_data.text2
        
        result = False
        if input_data.task == ComparisonTask.IDENTICAL:
            result = text1 == text2
        elif input_data.task == ComparisonTask.CONTAINS:
            result = text2 in text1 or text1 in text2
        elif input_data.task == ComparisonTask.PREFIX:
            result = text1.startswith(text2) or text2.startswith(text1)
        elif input_data.task == ComparisonTask.SUFFIX:
            result = text1.endswith(text2) or text2.endswith(text1)
        elif input_data.task == ComparisonTask.CASE_INSENSITIVE:
            result = text1.lower() == text2.lower()
        
        return AppOutput(result=result)

    async def unload(self):
        """Clean up resources here."""
        pass
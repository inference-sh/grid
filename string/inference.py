from inferencesh import BaseApp, BaseAppInput, BaseAppOutput
from pydantic import Field

class AppInput(BaseAppInput):
    text_input: str = Field(description="Text input string")

class AppOutput(BaseAppOutput):
    text_output: str = Field(description="Text output string")


class App(BaseApp):
    async def setup(self):
        pass

    async def run(self, input_data: AppInput) -> AppOutput:
        """Run prediction on the input data."""
        # Simply return the input text as output
        return AppOutput(
            text_output=input_data.text_input
        )

    async def unload(self):
        """Clean up resources here."""
        pass
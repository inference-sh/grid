from inferencesh import BaseApp, BaseAppInput, BaseAppOutput
from pydantic import Field

class AppInput(BaseAppInput):
    int_input: int = Field(description="Number input")

class AppOutput(BaseAppOutput):
    int_output: int = Field(description="Number output")


class App(BaseApp):
    async def setup(self):
        pass

    async def run(self, input_data: AppInput) -> AppOutput:
        """Run prediction on the input data."""
        # Simply return the input number as output
        return AppOutput(
            int_output=input_data.int_input
        )

    async def unload(self):
        """Clean up resources here."""
        pass
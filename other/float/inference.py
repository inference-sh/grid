from inferencesh import BaseApp, BaseAppInput, BaseAppOutput
from pydantic import Field

class AppInput(BaseAppInput):
    float_input: float = Field(description="Number input")

class AppOutput(BaseAppOutput):
    float_output: float = Field(description="Number output")


class App(BaseApp):
    async def setup(self, metadata):
        pass

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Run prediction on the input data."""
        # Simply return the input number as output
        return AppOutput(
            float_output=input_data.float_input
        )

    async def unload(self):
        """Clean up resources here."""
        pass
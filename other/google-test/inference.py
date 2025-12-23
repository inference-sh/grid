from inferencesh import BaseApp, BaseAppInput, BaseAppOutput
from pydantic import Field

class AppInput(BaseAppInput):
    pass

class AppOutput(BaseAppOutput):
    success: bool = Field(description="Whether the operation was successful")

class App(BaseApp):
    async def setup(self, metadata):
        """Initialize your model and resources here."""
       
        pass

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Run prediction on the input data."""

        return AppOutput(
           success=True
        )
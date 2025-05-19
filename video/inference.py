from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field

class AppInput(BaseAppInput):
    image_input: File = Field(description="Image input")

class AppOutput(BaseAppOutput):
    image_output: File = Field(description="Image output")


class App(BaseApp):
    async def setup(self):
        pass

    async def run(self, input_data: AppInput) -> AppOutput:
        """Run prediction on the input data."""
        # Check if input file exists and is accessible
        if not input_data.image_input.exists():
            raise RuntimeError(f"Input file does not exist at path: {input_data.image_input.path}")
        
        # Simply return the input image as output
        return AppOutput(
            image_output=input_data.image_input
        )

    async def unload(self):
        """Clean up resources here."""
        pass
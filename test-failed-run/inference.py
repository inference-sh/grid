from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File

class AppInput(BaseAppInput):
    text: str

class AppOutput(BaseAppOutput):
    result: File

class App(BaseApp):
    async def setup(self):
        """Initialize your model and resources here."""
        pass

    async def run(self, input_data: AppInput) -> AppOutput:
        """Run prediction on the input data."""
        # Process the input text and save result to file
        raise Exception("Failed to run")

    async def unload(self):
        """Clean up resources here."""
        pass
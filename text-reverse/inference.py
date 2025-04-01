from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File

class AppInput(BaseAppInput):
    text: str

class AppOutput(BaseAppOutput):
    text: str

class App(BaseApp):
    async def setup(self):
        """Initialize your model and resources here."""
        pass

    async def run(self, input_data: AppInput) -> AppOutput:
        """Run prediction on the input data."""

        return AppOutput(text=input_data.text[::-1])

    async def unload(self):
        """Clean up resources here."""
        pass
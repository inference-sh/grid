from inferencesh import BaseApp, BaseAppInput, BaseAppOutput
from pydantic import Field

class AppInput(BaseAppInput):
    input: str = Field(description="the input to the model")
class AppOutput(BaseAppOutput):
    output: str = Field(description="the output of the model")
    config: dict = Field(description="the config of the model")


class App(BaseApp):
    async def setup(self, metadata):
        self.variant = metadata.app_variant
        if self.variant == "default":
            self.config = {
                "model": "default-model",
            }
        elif self.variant == "variant1":
            self.config = {
                "model": "variant1-model",
            }       
        pass

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        return AppOutput(output=input_data.input, config=self.config)
        

    async def unload(self):
        """Clean up resources here."""
        pass
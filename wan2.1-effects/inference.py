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
        output_path = "/tmp/result.txt"
        with open(output_path, "w") as f:
            f.write(f"Processed: {input_data.text}")
        
        return AppOutput(result=File(path=output_path))

    async def unload(self):
        """Clean up resources here."""
        pass


    # https://github.com/huggingface/diffusers/issues/11222
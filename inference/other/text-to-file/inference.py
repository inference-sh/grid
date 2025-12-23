from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field

class AppInput(BaseAppInput):
    content: str = Field(description="The text content to write to the file")
    filename: str = Field(description="The name of the output file")

class AppOutput(BaseAppOutput):
    file: File = Field(description="The generated file containing the text content")

class App(BaseApp):
    async def setup(self, metadata):
        """Initialize your model and resources here."""
      
        pass

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Write the provided content to a file with the specified filename."""
        # Create the output file path in /tmp directory
        output_path = f"/tmp/{input_data.filename}"
        
        # Write the content to the file
        with open(output_path, "w") as f:
            f.write(input_data.content)
        
        # Return the file output
        # The File object will be automatically handled by the engine
        # which will upload the file to the configured storage backend
        return AppOutput(
            file=File(path=output_path)
        )

    async def unload(self):
        """Clean up resources here."""
        pass
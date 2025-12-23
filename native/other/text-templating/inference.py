from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, OutputMeta, TextMeta
from pydantic import Field

class AppInput(BaseAppInput):
    template: str = Field(description="The text template with placeholders like {1}, {2}, etc.")
    strings: list[str] = Field(description="List of strings to substitute into the template")

class AppOutput(BaseAppOutput):
    result: str = Field(description="The processed template with strings substituted")
    
class App(BaseApp):
    async def setup(self, metadata):
        """Initialize your model and resources here."""
        pass

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Process the template with the provided strings."""
        # Create a dictionary of replacements
        replacements = {str(i+1): string for i, string in enumerate(input_data.strings)}
        
        # Process the template
        result = input_data.template
        for placeholder, value in replacements.items():
            result = result.replace(f"{{{placeholder}}}", value)
            
        output_meta = OutputMeta(
            inputs=[TextMeta(tokens=len(input_data.template))],
            outputs=[TextMeta(tokens=len(result))]
        )
        
        return AppOutput(result=result, output_meta=output_meta)

    async def unload(self):
        """Clean up resources here."""
        pass
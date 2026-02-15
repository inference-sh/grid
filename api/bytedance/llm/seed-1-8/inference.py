from inferencesh import BaseApp, BaseAppSetup, File, OutputMeta, TextMeta
from pydantic import BaseModel, Field
from typing import List, Optional

# this is openai api schema so could we use openrouter chat builder from our sdk?
# curl https://ark.ap-southeast.bytepluses.com/api/v3/chat/completions \
#   -H "Content-Type: application/json" \
#   -H "Authorization: Bearer $ARK_API_KEY" \
#   -d $'{
#     "model": "seed-1-8-251228",
#     "messages": [
#         {
#             "content": [
#                 {
#                     "image_url": {
#                         "url": "https://ark-doc.tos-ap-southeast-1.bytepluses.com/see_i2v.jpeg"
#                     },
#                     "type": "image_url"
#                 },
#                 {
#                     "text": "What is the person doing in this picture?",
#                     "type": "text"
#                 }
#             ],
#             "role": "user"
#         }
#     ]
# }'

# Setup data is used to initialize the app and is passed to the setup method.
# Use this to define parameters that customize how your app starts up,
# such as model variants, precision settings, or feature toggles.
class AppSetup(BaseAppSetup):
    """Define your setup schema here.
    
    Setup parameters are provided once when the app instance starts.
    Use them for configuration that affects initialization, not per-request data.
    """
    model_variant: str = Field(default="base", description="Model variant to load (base, large, xl)")
    use_fp16: bool = Field(default=True, description="Use half precision for faster inference")
    enable_cache: bool = Field(default=False, description="Enable result caching")

class RunInput(BaseModel):
    """Define your input schema here.
    
    Use Field(description="...") to document each field - these descriptions
    are used to generate the API documentation and UI forms.
    """
    string_input: str = Field(description="The input string to process")
    number_input: int = Field(description="The input number to process")
    boolean_input: bool = Field(description="The input boolean to process")
    list_input: List[int] = Field(description="The input list to process")
    optional_input: Optional[str] = Field(None, description="The optional input to process")
    file_input: File = Field(description="The input file to process")

class RunOutput(BaseModel):
    """Define your output schema here.
    
    The output_meta field is handled by the SDK for usage-based pricing.
    """
    file_output: File = Field(description="an output file")
    string_output: str = Field(description="an output string")
    number_output: int = Field(description="an output number")
    boolean_output: bool = Field(description="an output boolean")
    list_output: List[int] = Field(description="an output list")
    optional_output: Optional[str] = Field(None, description="an optional output")


class App(BaseApp):
    
    async def setup(self, config: AppSetup):
        """Initialize your model and resources here.
        
        This method is called once when the app starts. Use it to:
        - Load ML models based on config parameters
        - Initialize tokenizers
        - Set up caches or connections
        - Access secrets via os.environ
        
        Args:
            config: Configuration parameters from AppSetup schema
        """
        # Store setup parameters for use in run()
        self.model_variant = config.model_variant
        self.use_fp16 = config.use_fp16
        
        # Initialize cache if enabled
        if config.enable_cache:
            self.cache = {}
        else:
            self.cache = None

    async def run(self, input_data: RunInput) -> RunOutput:
        """Run prediction on the input data.
        
        This method is called for each request. Use resources initialized in setup().
        
        Args:
            input_data: The validated input from the user
            
        Returns:
            RunOutput with your results
        """
        # Check if input file exists and is accessible
        if not input_data.file_input.exists():
            raise RuntimeError(f"Input file does not exist at path: {input_data.file_input.path}")
        
        # File metadata is populated automatically after file is downloaded/resolved
        # Available properties:
        # - input_data.file_input.uri: Original location (URL or file path)
        # - input_data.file_input.path: Resolved local file path
        # - input_data.file_input.content_type: MIME type of the file
        # - input_data.file_input.size: File size in bytes

        # Process inputs (these are just examples - replace with your logic)
        processed_string = input_data.string_input.upper()
        processed_number = input_data.number_input * 2
        processed_boolean = not input_data.boolean_input
        processed_list = [x + 1 for x in input_data.list_input]
        processed_optional = input_data.optional_input.lower() if input_data.optional_input else None

        # Write results to output file
        output_path = "/tmp/result.txt"
        with open(output_path, "w") as f:
            f.write(f"Processed: {input_data.string_input}")
            
        # Return output - the File will be automatically uploaded
        return RunOutput(
            file_output=File(path=output_path),
            string_output=processed_string,
            number_output=processed_number,
            boolean_output=processed_boolean,
            list_output=processed_list,
            optional_output=processed_optional
        )

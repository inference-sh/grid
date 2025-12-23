import os
import sys
import tempfile
import torch
from accelerate import Accelerator
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from typing import Optional

# Add current directory to Python path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Enable faster downloads globally
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

class AppInput(BaseAppInput):
    prompt: str = Field(description="Text prompt for image generation")
    content_image: Optional[File] = Field(None, description="Content reference image for subject/identity-driven generation")
    style_image: Optional[File] = Field(None, description="Style reference image for style transfer")
    extra_style_image: Optional[File] = Field(None, description="Extra style reference image (experimental feature)")
    width: Optional[int] = Field(None, description="Generation width (512-1536, step 16). If not specified, inferred from input image", ge=512, le=1536)
    height: Optional[int] = Field(None, description="Generation height (512-1536, step 16). If not specified, inferred from input image", ge=512, le=1536)
    num_steps: int = Field(default=25, description="Number of inference steps (1-50)", ge=1, le=50)
    guidance: float = Field(default=4.0, description="Guidance scale (1.0-5.0)", ge=1.0, le=5.0)
    seed: int = Field(default=-1, description="Random seed (-1 for random)")
    keep_size: bool = Field(default=False, description="Keep input image size (for style editing)")
    content_long_size: int = Field(default=512, description="Content reference image size (0-1024)", ge=0, le=1024)

class AppOutput(BaseAppOutput):
    generated_image: File = Field(description="Generated stylized image")




class App(BaseApp):
    def __init__(self):
        super().__init__()
        self.pipeline = None
        self.siglip_model = None
        self.siglip_processor = None
        self.accelerator = None
        self.device = None

    async def setup(self, metadata):
        """Initialize USO pipeline and models"""
        from uso.flux.pipeline import USOPipeline
        from transformers import SiglipVisionModel, SiglipImageProcessor
        
        # Initialize accelerator for proper device management
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        
        # Get variant from metadata (not environment variables!)
        variant = getattr(metadata, "app_variant", "default")
        
        # Map variant key to model type
        if variant == "default":
            model_type = "flux-dev"  # Default uses flux-dev
        elif variant == "flux-dev-fp8":
            model_type = "flux-dev-fp8"
        elif variant == "flux-schnell":
            model_type = "flux-schnell"
        elif variant == "flux-krea-dev":
            model_type = "flux-krea-dev"
        else:
            # Fallback to default
            model_type = "flux-dev"
            
        offload = False
        
        # Initialize USO pipeline
        self.pipeline = USOPipeline(
            model_type, 
            str(self.device), 
            offload, 
            only_lora=True, 
            lora_rank=128, 
            hf_download=True
        )
        
        # Load SigLIP vision encoder
        self.siglip_processor = SiglipImageProcessor.from_pretrained(
            "google/siglip-so400m-patch14-384"
        )
        self.siglip_model = SiglipVisionModel.from_pretrained(
            "google/siglip-so400m-patch14-384"
        )
        self.siglip_model.eval()
        self.siglip_model.to(self.device)
        
        # Attach vision encoder to pipeline
        self.pipeline.model.vision_encoder = self.siglip_model
        self.pipeline.model.vision_encoder_processor = self.siglip_processor

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Generate stylized image using USO pipeline"""
        from PIL import Image
        
        try:
            # Prepare inputs
            prompt = input_data.prompt
            
            # Load input images if provided
            content_img = None
            style_img = None
            extra_style_img = None
            reference_img = None  # For dimension inference
            
            if input_data.content_image and input_data.content_image.exists():
                content_img = Image.open(input_data.content_image.path).convert("RGB")
                reference_img = content_img  # Use content image for dimension inference
                
            if input_data.style_image and input_data.style_image.exists():
                style_img = Image.open(input_data.style_image.path).convert("RGB")
                if reference_img is None:  # Use style image if no content image
                    reference_img = style_img
                
            if input_data.extra_style_image and input_data.extra_style_image.exists():
                extra_style_img = Image.open(input_data.extra_style_image.path).convert("RGB")
                if reference_img is None:  # Use extra style image as last resort
                    reference_img = extra_style_img
            
            # Determine output dimensions
            if input_data.width is not None and input_data.height is not None:
                # Use specified dimensions
                output_width = input_data.width
                output_height = input_data.height
            elif reference_img is not None:
                # Infer from reference image, ensuring dimensions are within bounds and divisible by 16
                img_width, img_height = reference_img.size
                
                # Clamp to valid range
                output_width = max(512, min(1536, img_width))
                output_height = max(512, min(1536, img_height))
                
                # Round to nearest multiple of 16 (required by most diffusion models)
                output_width = (output_width // 16) * 16
                output_height = (output_height // 16) * 16
            else:
                # Default fallback when no images provided
                output_width = 1024
                output_height = 1024
            
            # Prepare generation arguments
            generation_args = [
                prompt,
                content_img,
                style_img, 
                extra_style_img,
                input_data.seed,
                output_width,
                output_height,
                input_data.guidance,
                input_data.num_steps,
                input_data.keep_size,
                input_data.content_long_size
            ]
            
            # Generate image using USO pipeline
            generated_image, download_path = self.pipeline.gradio_generate(*generation_args)
            
            # Create temporary output file if download_path is not provided
            if download_path is None:
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    output_path = tmp.name
                    generated_image.save(output_path, "PNG")
            else:
                output_path = download_path
            
            return AppOutput(generated_image=File(path=output_path))
            
        except Exception as e:
            raise ValueError(f"Failed to generate image: {str(e)}")

    async def unload(self):
        """Clean up GPU memory and resources"""
        if hasattr(self, 'pipeline') and self.pipeline is not None:
            del self.pipeline
        if hasattr(self, 'siglip_model') and self.siglip_model is not None:
            del self.siglip_model
        if hasattr(self, 'siglip_processor') and self.siglip_processor is not None:
            del self.siglip_processor
            
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        import gc
        gc.collect()
import os
import torch
import tempfile
from PIL import Image
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from typing import List, Optional
from diffusers import QwenImageEditPlusPipeline, FlowMatchEulerDiscreteScheduler
from accelerate import Accelerator

# Enable faster HuggingFace downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# App-wide configuration
class AppConfig:
    """Global app configuration and defaults."""
    DEFAULT_WIDTH = 1024
    DEFAULT_HEIGHT = 1024
    MIN_DIMENSION = 256
    MAX_DIMENSION = 2048
    MODEL_ID = "Qwen/Qwen-Image-Edit-2509"
    DEFAULT_STEPS = 30  # Standard model works well with 30 steps

class AppInput(BaseAppInput):
    images: List[File] = Field(description="Input images (1-3 images) for editing")
    prompt: str = Field(description="Text prompt describing the desired edit")
    negative_prompt: Optional[str] = Field(default=" ", description="Negative prompt to guide what to avoid")
    width: Optional[int] = Field(
        default=None,
        ge=AppConfig.MIN_DIMENSION,
        le=AppConfig.MAX_DIMENSION,
        description="Output image width in pixels (defaults to first input image width if not specified)"
    )
    height: Optional[int] = Field(
        default=None,
        ge=AppConfig.MIN_DIMENSION,
        le=AppConfig.MAX_DIMENSION,
        description="Output image height in pixels (defaults to first input image height if not specified)"
    )
    num_inference_steps: int = Field(default=AppConfig.DEFAULT_STEPS, ge=1, le=100, description="Number of denoising steps (30 recommended for quality/speed balance)")
    guidance_scale: float = Field(default=1.0, ge=0.0, le=10.0, description="Guidance scale for generation")
    true_cfg_scale: float = Field(default=1.0, ge=0.0, le=10.0, description="True CFG scale for the model")
    seed: int = Field(default=42, description="Random seed for reproducibility")

class AppOutput(BaseAppOutput):
    image: File = Field(description="The edited output image")

class App(BaseApp):
    async def setup(self, metadata):
        """Initialize the Qwen Image Edit Plus pipeline (standard version)."""
        # Setup device using accelerate
        self.accelerator = Accelerator()
        self.device = self.accelerator.device

        print(f"🔧 Setting up Qwen Image Edit Plus pipeline on device: {self.device}")

        # Load scheduler
        scheduler_config = {
            "num_train_timesteps": 1000,
            "shift": 3.0,
        }
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

        # Load pipeline using app-wide config
        print(f"📦 Loading {AppConfig.MODEL_ID} model...")
        self.pipeline = QwenImageEditPlusPipeline.from_pretrained(
            AppConfig.MODEL_ID,
            scheduler=scheduler,
            torch_dtype=torch.bfloat16
        )
        self.pipeline.to(self.device)

        print("✅ Pipeline setup complete!")

    def _load_images(self, image_files: List[File]) -> List[Image.Image]:
        """Load PIL Images from File objects."""
        images = []
        for img_file in image_files:
            if not img_file.exists():
                raise RuntimeError(f"Input image does not exist at path: {img_file.path}")
            img = Image.open(img_file.path).convert("RGB")
            images.append(img)
        return images

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Process images with the Qwen Image Edit Plus pipeline."""
        # Validate input images (1-3 images)
        if not input_data.images or len(input_data.images) == 0:
            raise ValueError("At least one input image is required")
        if len(input_data.images) > 3:
            raise ValueError("Maximum 3 input images are supported")

        print(f"🎨 Processing {len(input_data.images)} image(s) with prompt: '{input_data.prompt}'")

        # Load images
        images = self._load_images(input_data.images)

        # Use first input image dimensions if width/height not specified
        width = input_data.width
        height = input_data.height

        if width is None or height is None:
            first_image = images[0]
            original_width = first_image.width
            original_height = first_image.height

            # Calculate scaled dimensions maintaining aspect ratio
            # Scale down max dimension to 1024 if larger
            max_dimension = max(original_width, original_height)

            if max_dimension > AppConfig.DEFAULT_WIDTH:
                scale_factor = AppConfig.DEFAULT_WIDTH / max_dimension
                scaled_width = int(original_width * scale_factor)
                scaled_height = int(original_height * scale_factor)
            else:
                scaled_width = original_width
                scaled_height = original_height

            # Use calculated dimensions if not specified
            if width is None:
                width = scaled_width
            if height is None:
                height = scaled_height

            if (scaled_width, scaled_height) != (original_width, original_height):
                print(f"📐 Scaled from {original_width}x{original_height} to {width}x{height} (maintaining aspect ratio, max dimension: {AppConfig.DEFAULT_WIDTH})")
            else:
                print(f"📐 Using original dimensions: {width}x{height} (no scaling needed)")
        else:
            print(f"📐 Using specified dimensions: {width}x{height}")

        # Prepare pipeline inputs
        inputs = {
            "image": images,
            "prompt": input_data.prompt,
            "width": width,
            "height": height,
            "generator": torch.manual_seed(input_data.seed),
            "true_cfg_scale": input_data.true_cfg_scale,
            "negative_prompt": input_data.negative_prompt,
            "num_inference_steps": input_data.num_inference_steps,
            "guidance_scale": input_data.guidance_scale,
            "num_images_per_prompt": 1,
        }

        # Run inference
        print(f"🚀 Running inference with {input_data.num_inference_steps} steps...")
        with torch.inference_mode():
            output = self.pipeline(**inputs)

        # Save output image
        output_image = output.images[0]
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            output_path = tmp.name
        output_image.save(output_path, format='PNG')

        print(f"✅ Image editing complete!")

        return AppOutput(image=File(path=output_path))


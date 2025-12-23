from typing import Optional, List
from pydantic import Field
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from PIL import Image
import torch
from .InstantCharacter.pipeline import InstantCharacterFluxPipeline
from huggingface_hub import snapshot_download
import os
from enum import Enum
import glob

class StyleType(str, Enum):
    NONE = "none"
    GHIBLI = "ghibli"
    MAKOTO_SHINKAI = "makoto_shinkai"

    @property
    def lora_path(self) -> Optional[str]:
        if self == StyleType.NONE:
            return None
        elif self == StyleType.GHIBLI:
            return "checkpoints/style_lora/ghibli_style.safetensors"
        elif self == StyleType.MAKOTO_SHINKAI:
            return "checkpoints/style_lora/Makoto_Shinkai_style.safetensors"

    @property
    def repo_id(self) -> Optional[str]:
        if self == StyleType.NONE:
            return None
        elif self == StyleType.GHIBLI:
            return "InstantX/FLUX.1-dev-LoRA-Ghibli"
        elif self == StyleType.MAKOTO_SHINKAI:
            return "InstantX/FLUX.1-dev-LoRA-Makoto-Shinkai"

class AppInput(BaseAppInput):
    prompt: str = Field(
        ...,
        description="The text prompt to generate the image from",
        examples=["A girl is playing a guitar in street"]
    )
    num_inference_steps: int = Field(
        28,
        description="Number of denoising steps",
        ge=1,
        le=100
    )
    guidance_scale: float = Field(
        3.5,
        description="Classifier-free guidance scale",
        ge=1.0,
        le=20.0
    )
    width: int = Field(
        1024,
        description="Width of the generated image",
        ge=256,
        le=2048
    )
    height: int = Field(
        1024,
        description="Height of the generated image",
        ge=256,
        le=2048
    )
    subject_image: File = Field(
        ...,
        description="Reference image for character preservation"
    )
    subject_scale: float = Field(
        0.9,
        description="Scale of the subject in the generated image",
        ge=0.0,
        le=1.0
    )
    style: StyleType = Field(
        StyleType.NONE,
        description="Style to apply to the generated image"
    )
    style_trigger: Optional[str] = Field(
        None,
        description="Trigger word for the style LoRA (e.g., 'ghibli style' or 'Makoto Shinkai style')"
    )

class AppOutput(BaseAppOutput):
    result: File

class App(BaseApp):
    pipeline: Optional[InstantCharacterFluxPipeline] = None

    async def setup(self, metadata):
        """Initialize the InstantCharacter model."""
        # Create checkpoints directory if it doesn't exist
        os.makedirs('checkpoints', exist_ok=True)

        # Download model files
        print("Downloading InstantCharacter model files...")
        snapshot_download(
            repo_id="Tencent/InstantCharacter",
            local_dir="checkpoints",
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print("Model files downloaded successfully")

        # Initialize pipeline components
        ip_adapter_path = 'checkpoints/instantcharacter_ip-adapter.bin'
        base_model = 'black-forest-labs/FLUX.1-dev'
        image_encoder_path = 'google/siglip-so400m-patch14-384'
        image_encoder_2_path = 'facebook/dinov2-giant'

        # Initialize the pipeline
        self.pipeline = InstantCharacterFluxPipeline.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16
        )
        self.pipeline.to("cuda")

        # Initialize the adapter
        self.pipeline.init_adapter(
            image_encoder_path=image_encoder_path,
            image_encoder_2_path=image_encoder_2_path,
            subject_ipadapter_cfg=dict(
                subject_ip_adapter_path=ip_adapter_path,
                nb_token=1024
            ),
        )

    def _check_lora_directory(self):
        """Check and print the contents of the LoRA directory."""
        lora_dir = 'checkpoints/style_lora'
        print("\nChecking LoRA directory contents:")
        print(f"Directory path: {os.path.abspath(lora_dir)}")
        print(f"Directory exists: {os.path.exists(lora_dir)}")
        
        if os.path.exists(lora_dir):
            print("\nFiles in directory:")
            for file in os.listdir(lora_dir):
                file_path = os.path.join(lora_dir, file)
                print(f"- {file} (exists: {os.path.exists(file_path)}, size: {os.path.getsize(file_path) if os.path.exists(file_path) else 'N/A'} bytes)")
            
            print("\nSearching for .safetensors files:")
            safetensors_files = glob.glob(os.path.join(lora_dir, "*.safetensors"))
            for file in safetensors_files:
                print(f"- {os.path.basename(file)} (size: {os.path.getsize(file)} bytes)")

    async def _download_style_lora(self, style: StyleType) -> None:
        """Download the specified style LoRA if it doesn't exist."""
        if style == StyleType.NONE:
            return

        # Create style_lora directory if it doesn't exist
        os.makedirs('checkpoints/style_lora', exist_ok=True)

        # Check if LoRA file already exists
        if os.path.exists(style.lora_path):
            return

        # Check directory contents before download
        self._check_lora_directory()

        # Download the LoRA file
        print(f"\nDownloading {style.value} style LoRA...")
        print(f"Repository ID: {style.repo_id}")
        print(f"Expected file path: {style.lora_path}")
        
        snapshot_download(
            repo_id=style.repo_id,
            local_dir='checkpoints/style_lora',
            local_dir_use_symlinks=False,
            resume_download=True
        )
        
        # Check directory contents after download
        self._check_lora_directory()
        print(f"{style.value} style LoRA downloaded successfully")

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Generate an image based on the input prompt and reference image."""
        if not self.pipeline:
            raise RuntimeError("Model not initialized. Call setup() first.")

        # Download style LoRA if needed
        await self._download_style_lora(input_data.style)

        # Load the reference image
        ref_image = Image.open(input_data.subject_image.path).convert('RGB')

        # Generate the image
        generator = torch.Generator('cuda').manual_seed(torch.randint(0, 2147483647, (1,)).item())

        if input_data.style != StyleType.NONE:
            # Use style LoRA
            if not input_data.style_trigger:
                raise ValueError("style_trigger is required when using a style")
            
            print("\nLoRA generation parameters:")
            print(f"LoRA path: {input_data.style.lora_path}")
            print(f"File exists: {os.path.exists(input_data.style.lora_path)}")
            print(f"Style trigger: {input_data.style_trigger}")
            print(f"Prompt: {input_data.prompt}")
            print(f"Num inference steps: {input_data.num_inference_steps}")
            print(f"Guidance scale: {input_data.guidance_scale}")
            print(f"Subject scale: {input_data.subject_scale}")

            image = self.pipeline.with_style_lora(
                lora_file_path=input_data.style.lora_path,
                trigger=input_data.style_trigger,
                prompt=input_data.prompt,
                num_inference_steps=input_data.num_inference_steps,
                guidance_scale=input_data.guidance_scale,
                subject_image=ref_image,
                subject_scale=input_data.subject_scale,
                generator=generator,
            ).images[0]
        else:
            # Generate without style
            image = self.pipeline(
                prompt=input_data.prompt,
                num_inference_steps=input_data.num_inference_steps,
                guidance_scale=input_data.guidance_scale,
                subject_image=ref_image,
                subject_scale=input_data.subject_scale,
                generator=generator,
            ).images[0]

        # Save the image
        output_path = "/tmp/generated_image.png"
        image.save(output_path)

        return AppOutput(result=File(path=output_path))

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from typing import Optional
from pydantic import Field

import torch
torch.set_float32_matmul_precision("high")

from diffusers import FluxPipeline, FluxTransformer2DModel, TorchAoConfig
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from PIL import Image

# Quantization configurations for different variants
configs = {
    "default": {
        "quantization": None,
        "compile": False,
        "fuse_attn_projections": False,
        "cpu_offload": False,
    },
    "bf16-cpu-offload": {
        "quantization": None,
        "compile": False,
        "fuse_attn_projections": False,
        "cpu_offload": True,
    },
    "int8dq": {
        "quantization": "int8dq",
        "compile": False,
        "fuse_attn_projections": False,
        "cpu_offload": False,
    },
    "int8wo": {
        "quantization": "int8wo",
        "compile": False,
        "fuse_attn_projections": False,
        "cpu_offload": False,
    },
    "int4wo": {
        "quantization": "int4wo",
        "compile": False,
        "fuse_attn_projections": False,
        "cpu_offload": False,
    },
    "fp8dq": {
        "quantization": "float8dq",
        "compile": False,
        "fuse_attn_projections": False,
        "cpu_offload": False,
    },
    "fp8wo": {
        "quantization": "float8wo",
        "compile": False,
        "fuse_attn_projections": False,
        "cpu_offload": False,
    },
    "fp8dqrow": {
        "quantization": "float8dq_e4m3_row",
        "compile": False,
        "fuse_attn_projections": False,
        "cpu_offload": False,
    },
    "autoquant": {
        "quantization": "autoquant",
        "compile": False,
        "fuse_attn_projections": False,
        "cpu_offload": False,
    }
}

class AppInput(BaseAppInput):
    prompt: str = Field(
        description="The text prompt describing the desired image",
        examples=["a photo of a cat sitting on a windowsill at sunset", 
                 "an oil painting of mountains at sunrise"],
        min_length=1,
        max_length=1000,
    )
    
    width: int = Field(
        default=1024,
        description="Width of generated image in pixels",
        ge=512,
        le=2048,
        examples=[1024, 768]
    )
    
    height: int = Field(
        default=1024,
        description="Height of generated image in pixels",
        ge=512, 
        le=2048,
        examples=[1024, 768]
    )
    
    number_of_steps: int = Field(
        default=28,
        description="Number of denoising steps (higher = better quality but slower)",
        ge=1,
        le=100,
        examples=[28, 50]
    )
    
    guidance_scale: float = Field(
        default=4.0,
        description="How closely to follow the prompt (higher = more faithful but less creative)",
        ge=1.0,
        le=20.0,
        examples=[4.0, 7.5]
    )
    
    seed: int = Field(
        default=0,
        description="Random seed for reproducible results",
        ge=0,
        examples=[42, 123456]
    )

    class Config:
        json_schema_extra = {
            "title": "Text to Image Generation Input",
            "description": "Input parameters for generating images from text descriptions",
            "examples": [{
                "prompt": "a photo of a cat sitting on a windowsill at sunset",
                "number_of_images": 1,
                "width": 1024,
                "height": 1024,
                "number_of_steps": 28,
                "guidance_scale": 4.0,
                "seed": 42
            }]
        }

class AppOutput(BaseAppOutput):
    image: File = Field(
        description="Generated image file"
    )

class App(BaseApp):
    async def setup(self, metadata):
        """Initialize FLUX model and set optimizations"""
        self.device = "cuda"
        
        model_id = "black-forest-labs/Flux.1-Dev"
        dtype = torch.bfloat16

        # Get variant configuration
        self.variant_config = configs[metadata.app_variant]
        
        # Initialize quantization config if needed
        quantization_config = None
        if self.variant_config["quantization"]:
            quantization_config = TorchAoConfig(self.variant_config["quantization"])

        # Load transformer with quantization if specified
        transformer = FluxTransformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            quantization_config=quantization_config,
            torch_dtype=dtype,
        )

        # Apply compilation if enabled
        if self.variant_config["compile"]:
            transformer = torch.compile(transformer, mode="max-autotune", fullgraph=True)

        # Create pipeline with quantized transformer
        self.pipe = FluxPipeline.from_pretrained(
            model_id,
            transformer=transformer,
            torch_dtype=dtype,
        )
        if self.variant_config["cpu_offload"]:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(self.device)

        # Apply attention projection fusion if enabled
        if self.variant_config["fuse_attn_projections"]:
            self.pipe.transformer.fuse_qkv_projections()

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Run text to image with FLUX"""
        images = self.pipe(
                prompt=input_data.prompt,
                num_images_per_prompt=1,
                width=input_data.width,
                height=input_data.height,
                guidance_scale=input_data.guidance_scale,
                num_inference_steps=input_data.number_of_steps,
                max_sequence_length=512,
                generator=torch.Generator(self.device).manual_seed(input_data.seed)
            ).images

        output_path = f"/tmp/output_{hash(str(torch.rand(1)[0].item()))}.png"
        images[0].save(output_path)
        return AppOutput(image=File.from_path(output_path))

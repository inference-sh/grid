import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import urllib.request
from io import BytesIO
from typing import Optional
from pydantic import Field

import torch
torch.set_float32_matmul_precision("high")

from torchao.quantization import quantize_, autoquant

from diffusers import FluxPipeline
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from PIL import Image

def load_image_from_url_or_path(url_or_path: str) -> Image.Image:
    print(f"Loading image from URL or path: {url_or_path}")
    if url_or_path.startswith("http") or url_or_path.startswith("https"):
        headers = {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/91.0.4472.124 Safari/537.36'
            )
        }
        req = urllib.request.Request(url_or_path, headers=headers)
        return Image.open(BytesIO(urllib.request.urlopen(req).read()))
    else:
        return Image.open(url_or_path)
    
class AppInput(BaseAppInput):
    prompt: str = Field(
        description="The text prompt describing the desired image",
        examples=["a photo of a cat sitting on a windowsill at sunset", 
                 "an oil painting of mountains at sunrise"],
        min_length=1,
        max_length=1000,
    )

    number_of_images: int = Field(
        default=1,
        description="Number of images to generate", 
        ge=1,
        le=4,
        examples=[1, 2]
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
    result: list[File] = Field(
        description="List of generated image files",
        min_items=1
    )


def load_pipeline(
    pipeline: FluxPipeline,
    fuse_attn_projections: bool = False,
    compile: bool = True,
    quantization: str = "None",
    sparsify: bool = False,
    compile_vae: bool = False,
    # dtype: torch.dtype = torch.bfloat16,
    # device: str = "cuda",
) -> FluxPipeline:
    # pipeline = DiffusionPipeline.from_pretrained(ckpt_id, torch_dtype=dtype).to(device)

    if fuse_attn_projections:
        pipeline.transformer.fuse_qkv_projections()
        if compile_vae:
            pipeline.vae.fuse_qkv_projections()

    if quantization == "autoquant" and compile:
        pipeline.transformer.to(memory_format=torch.channels_last)
        pipeline.transformer = torch.compile(pipeline.transformer, mode="max-autotune", fullgraph=True)
        if compile_vae:
            pipeline.vae.to(memory_format=torch.channels_last)
            pipeline.vae.decode = torch.compile(pipeline.vae.decode, mode="max-autotune", fullgraph=True)

    if not sparsify:
        if quantization == "int8dq":
            from torchao.quantization import int8_dynamic_activation_int8_weight

            quantize_(pipeline.transformer, int8_dynamic_activation_int8_weight())
            if compile_vae:
                quantize_(pipeline.vae, int8_dynamic_activation_int8_weight())
        elif quantization == "int8wo":
            from torchao.quantization import int8_weight_only

            quantize_(pipeline.transformer, int8_weight_only())
            if compile_vae:
                quantize_(pipeline.vae, int8_weight_only())
        elif quantization == "int4wo":
            from torchao.quantization import int4_weight_only

            quantize_(pipeline.transformer, int4_weight_only())
            if compile_vae:
                quantize_(pipeline.vae, int4_weight_only())
        elif quantization == "fp6_e3m2":
            from torchao.quantization import fpx_weight_only

            quantize_(pipeline.transformer, fpx_weight_only(3, 2))
            if compile_vae:
                quantize_(pipeline.vae, fpx_weight_only(3, 2))

        elif quantization == "fp5_e2m2":
            from torchao.quantization import fpx_weight_only

            quantize_(pipeline.transformer, fpx_weight_only(2, 2))
            if compile_vae:
                quantize_(pipeline.vae, fpx_weight_only(2, 2))

        elif quantization == "fp4_e2m1":
            from torchao.quantization import fpx_weight_only

            quantize_(pipeline.transformer, fpx_weight_only(2, 1))
            if compile_vae:
                quantize_(pipeline.vae, fpx_weight_only(2, 1))
        elif quantization == "fp8wo":
            from torchao.quantization import float8_weight_only

            quantize_(pipeline.transformer, float8_weight_only())
            if compile_vae:
                quantize_(pipeline.vae, float8_weight_only())
        elif quantization == "fp8dq":
            from torchao.quantization import float8_dynamic_activation_float8_weight

            quantize_(pipeline.transformer, float8_dynamic_activation_float8_weight())
            if compile_vae:
                quantize_(pipeline.vae, float8_dynamic_activation_float8_weight())
        elif quantization == "fp8dqrow":
            from torchao.quantization import float8_dynamic_activation_float8_weight
            from torchao.quantization.quant_api import PerRow

            quantize_(pipeline.transformer, float8_dynamic_activation_float8_weight(granularity=PerRow()))
            if compile_vae:
                quantize_(pipeline.vae, float8_dynamic_activation_float8_weight(granularity=PerRow()))
        elif quantization == "autoquant":
            pipeline.transformer = autoquant(pipeline.transformer, error_on_unseen=False)
            if compile_vae:
                pipeline.vae = autoquant(pipeline.vae, error_on_unseen=False)

    if sparsify:
        from torchao.sparsity import sparsify_, int8_dynamic_activation_int8_semi_sparse_weight

        sparsify_(pipeline.transformer, int8_dynamic_activation_int8_semi_sparse_weight())
        if compile_vae:
            sparsify_(pipeline.vae, int8_dynamic_activation_int8_semi_sparse_weight())

    if quantization != "autoquant" and compile:
        pipeline.transformer.to(memory_format=torch.channels_last)
        pipeline.transformer = torch.compile(pipeline.transformer, mode="max-autotune", fullgraph=True)
        if compile_vae:
            pipeline.vae.to(memory_format=torch.channels_last)
            pipeline.vae.decode = torch.compile(pipeline.vae.decode, mode="max-autotune", fullgraph=True)

    # pipeline.set_progress_bar_config(disable=True)
    return pipeline

class App(BaseApp):
    async def setup(self):
        """Initialize FLUX model and set optimizations"""
        # Initialize FLUX
        self.device = "cuda"
        
        model_id = "black-forest-labs/Flux.1-Dev"
        dtype = torch.bfloat16

        # quantization_config = TorchAoConfig("fp8dq")
        # transformer = FluxTransformer2DModel.from_pretrained(
        #     model_id,
        #     subfolder="transformer",
        #     quantization_config=quantization_config,
        #     torch_dtype=dtype,
        # )
        
        pipe = FluxPipeline.from_pretrained(
            model_id,
            # transformer=transformer,
            torch_dtype=dtype,
        )
        pipe.to("cuda")

        self.pipe = load_pipeline(
            pipeline=pipe,
            fuse_attn_projections=True,
            # compile=True,
            # quantization="fp8dq",
            # sparsify=False,
            # dtype=torch.bfloat16,
            # device=self.device,
        )


    async def run(self, input_data: AppInput) -> AppOutput:
        """Run text to image with FLUX"""
        # Load input image

        images = self.pipe(
                prompt=input_data.prompt,
                num_images_per_prompt=input_data.number_of_images,
                width=input_data.width,
                height=input_data.height,
                guidance_scale=input_data.guidance_scale,
                num_inference_steps=input_data.number_of_steps,
                max_sequence_length=512,
                generator=torch.Generator(self.device).manual_seed(input_data.seed)
            ).images

        output_files = []
        for i, image in enumerate(images):
            output_path = f"/tmp/output_{i}_{hash(str(torch.rand(1)[0].item()))}.png"
            image.save(output_path)
            output_files.append(File.from_path(output_path))
        return AppOutput(result=output_files)

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
import torch
from PIL import Image
from .upscale import upscale, UpscaleMode, SeamFixMode
from typing import Optional
from io import BytesIO
from diffusers import FluxInpaintPipeline
from RealESRGAN import RealESRGAN
from huggingface_hub import hf_hub_download
import os
from pydantic import Field
# Ensure HF transfer is enabled

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

class AppInput(BaseAppInput):
    image: File = Field(description="The image to upscale")
    target_width: int = Field(
        description="The width of the output image", 
        default=2048,
        ge=512,
        le=4096,
        examples=[1024, 2048, 4096]
    )
    target_height: int = Field(
        description="The height of the output image",
        default=2048,
        ge=512, 
        le=4096,
        examples=[1024, 2048, 4096]
    )
    tile_width: int = Field(
        description="The width of the tile to upscale",
        default=1024,
        ge=256,
        le=2048,
        examples=[512, 1024, 2048]
    )
    tile_height: int = Field(
        description="The height of the tile to upscale",
        default=1024,
        ge=256,
        le=2048,
        examples=[512, 1024, 2048]
    )
    redraw_padding: int = Field(
        description="The padding to redraw",
        default=32,
        ge=0,
        le=128,
        examples=[16, 32, 64]
    )
    redraw_mask_blur: int = Field(
        description="The blur radius for the redraw mask",
        default=8,
        ge=0,
        le=32,
        examples=[4, 8, 16]
    )
    upscale_mode: UpscaleMode = Field(
        description="The mode to upscale the image",
        default=UpscaleMode.CHESS,
        enum=[UpscaleMode.LINEAR, UpscaleMode.CHESS, UpscaleMode.NONE]
    )
    seam_fix_mode: SeamFixMode = Field(
        description="The mode to fix the seams",
        default=SeamFixMode.NONE,
        enum=[SeamFixMode.NONE, SeamFixMode.BAND_PASS, SeamFixMode.HALF_TILE, SeamFixMode.HALF_TILE_PLUS_INTERSECTIONS]
    )
    prompt: str = Field(
        description="The prompt for the image",
        default="",
        examples=["enhance details, high quality", "sharp, clear, detailed"]
    )
    negative_prompt: str = Field(
        description="The negative prompt for the image",
        default="",
        examples=["blurry, low quality", "noise, artifacts"]
    )
    strength: float = Field(
        description="The strength of the prompt",
        default=0.3,
        ge=0.0,
        le=1.0,
        examples=[0.3, 0.5, 0.7]
    )
    guidance_scale: float = Field(
        description="The guidance scale for the image",
        default=7.5,
        ge=1.0,
        le=20.0,
        examples=[5.0, 7.5, 10.0]
    )
    seed: int = Field(
        description="The seed for the image",
        default=0,
        ge=0,
        examples=[42, 123456]
    )

class AppOutput(BaseAppOutput):
    result: File = Field(description="The upscaled image")

class App(BaseApp):
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    pipe: Optional[FluxInpaintPipeline] = None
    esrgan: Optional[RealESRGAN] = None

    async def setup(self, metadata):
        """Initialize FLUX model and RealESRGAN"""
        # Initialize FLUX
        model_id = "black-forest-labs/FLUX.1-dev"
        self.pipe = FluxInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16
        ).to(self.device)

        # Initialize RealESRGAN
        model_path = hf_hub_download(
            repo_id="ai-forever/Real-ESRGAN",
            filename="RealESRGAN_x4.pth"
        )
        self.esrgan = RealESRGAN(self.device, scale=4)
        self.esrgan.load_weights(model_path)

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Run upscaling with FLUX and RealESRGAN"""
        # Load input image
        image = Image.open(input_data.image.path)

        def upscale_fn(img: Image.Image, scale_factor: int) -> Image.Image:
            esrgan_result = self.esrgan.predict(img)
            return esrgan_result.resize(
                (img.width * scale_factor, img.height * scale_factor),
                Image.Resampling.LANCZOS
            )

        def process_fn(img: Image.Image, mask: Image.Image) -> Image.Image:
            return self.pipe(
                prompt=input_data.prompt,
                image=img,
                mask_image=mask,
                width=img.width,
                height=img.height,
                strength=input_data.strength,
                guidance_scale=input_data.guidance_scale,
                num_inference_steps=int(10/input_data.strength),
                max_sequence_length=512,
                generator=torch.Generator(self.device).manual_seed(input_data.seed)
            ).images[0]

        # Process the image using direct upscale function
        result = upscale(
            image=image,
            target_width=input_data.target_width,
            target_height=input_data.target_height,
            tile_width=input_data.tile_width,
            tile_height=input_data.tile_height,
            redraw_padding=input_data.redraw_padding,
            redraw_mask_blur=input_data.redraw_mask_blur,
            upscale_mode=input_data.upscale_mode,
            seam_fix_mode=input_data.seam_fix_mode,
            upscale_fn=upscale_fn,
            process_fn=process_fn
        )

        # Save and return result
        output_path = "/tmp/upscaled.png"
        result.save(output_path)
        return AppOutput(result=File.from_path(output_path))

    async def unload(self):
        """Clean up resources"""
        del self.pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
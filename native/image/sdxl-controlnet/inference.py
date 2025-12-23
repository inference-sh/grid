import os
import sys

# Add ControlNetPlus folder to Python path
controlnet_plus_path = os.path.join(os.path.dirname(__file__), "ControlNetPlus")
sys.path.append(controlnet_plus_path)

from typing import Optional, List
from pydantic import Field
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from diffusers import StableDiffusionXLPipeline, AutoencoderKL, EulerAncestralDiscreteScheduler
from controlnet_aux import (
    LineartAnimeDetector, LineartDetector, HEDdetector, MLSDdetector,
    MidasDetector, ZoeDetector
)
from .ControlNetPlus.models.controlnet_union import ControlNetModel_Union
from .ControlNetPlus.pipeline.pipeline_controlnet_union_sd_xl import StableDiffusionXLControlNetUnionPipeline
import torch
import cv2
import numpy as np
from PIL import Image
from enum import Enum

class ControlNetType(str, Enum):
    OPENPOSE = "openpose"
    DEPTH = "depth"
    HED = "hed"
    CANNY = "canny"
    LINEART = "lineart"
    ANIME_LINEART = "anime_lineart"
    MLSD = "mlsd"
    NORMAL = "normal"
    SEGMENT = "segment"

def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y

class ControlNetInput(BaseAppInput):
    type: ControlNetType = Field(
        ...,
        description="Type of ControlNet to use"
    )
    image: File = Field(
        ...,
        description="Input image for the ControlNet"
    )
    strength: float = Field(
        1.0,
        description="Strength of the ControlNet effect",
        ge=0.0,
        le=1.0
    )
    pre_process: bool = Field(
        True,
        description="Whether to pre-process the input image (e.g., apply canny/depth/hed/mlsd/normal/segment/openpose transformation). Set to False if the input is already pre-processed.",
    )

class AppInput(BaseAppInput):
    prompt: str = Field(
        ...,
        description="The text prompt to generate the image from",
        examples=["A majestic lion jumping from a big stone at night"]
    )
    negative_prompt: Optional[str] = Field(
        None,
        description="Negative prompt to avoid certain elements in the generated image",
        examples=["blurry, low quality, distorted"]
    )
    num_inference_steps: int = Field(
        50,
        description="Number of denoising steps",
        ge=1,
        le=100
    )
    guidance_scale: float = Field(
        7.5,
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
    model_url: str = Field(
        "stabilityai/stable-diffusion-xl-base-1.0",
        description="URL or path to a custom Stable Diffusion XL model",
        examples=["stabilityai/stable-diffusion-xl-base-1.0"]
    )
    controlnets: List[ControlNetInput] = Field(
        [],
        description="List of ControlNets to use for image generation"
    )

class AppOutput(BaseAppOutput):
    result: File

class App(BaseApp):
    pipeline: Optional[StableDiffusionXLControlNetUnionPipeline] = None
    default_model_url: str = "stabilityai/stable-diffusion-xl-base-1.0"
    processors: dict = {}

    async def setup(self, metadata):
        """Initialize the Stable Diffusion XL model with ControlNet support."""
        # Initialize processors
        self.processors = {
            "anime_lineart": LineartAnimeDetector.from_pretrained('lllyasviel/Annotators').to("cuda"),
            "lineart": LineartDetector.from_pretrained('lllyasviel/Annotators').to("cuda"),
            "hed": HEDdetector.from_pretrained('lllyasviel/Annotators').to("cuda"),
            "mlsd": MLSDdetector.from_pretrained('lllyasviel/Annotators').to("cuda"),
            "depth": {
                "zoe": ZoeDetector.from_pretrained("lllyasviel/Annotators").to("cuda"),
                "midas": MidasDetector.from_pretrained("lllyasviel/Annotators").to("cuda")
            }
        }

        # Initialize pipeline components
        eulera_scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
            self.default_model_url, subfolder="scheduler"
        )
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
        )
        controlnet_model = ControlNetModel_Union.from_pretrained(
            "xinsir/controlnet-union-sdxl-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True
        )

        self.pipeline = StableDiffusionXLControlNetUnionPipeline.from_pretrained(
            self.default_model_url,
            controlnet=controlnet_model,
            vae=vae,
            torch_dtype=torch.float16,
            scheduler=eulera_scheduler,
        )
        self.pipeline.to("cuda")

    def _process_controlnet_image(self, controlnet: ControlNetInput) -> Image.Image:
        """Process the input image according to the ControlNet type."""
        # Read the image
        img = cv2.imread(controlnet.image.path)
        
        # If pre_process is False, just resize the image and return
        if not controlnet.pre_process:
            height, width, _ = img.shape
            ratio = np.sqrt(1024. * 1024. / (width * height))
            new_width, new_height = int(width * ratio), int(height * ratio)
            processed_img = cv2.resize(img, (new_width, new_height))
            return Image.fromarray(processed_img)
        
        # Process based on ControlNet type
        if controlnet.type == ControlNetType.CANNY:
            processed_img = cv2.Canny(img, 100, 200)
            processed_img = HWC3(processed_img)
        elif controlnet.type == ControlNetType.DEPTH:
            # Randomly choose between Zoe and Midas
            processor = np.random.choice(list(self.processors["depth"].values()))
            processed_img = processor(img, output_type='cv2')
        elif controlnet.type in [ControlNetType.ANIME_LINEART, ControlNetType.LINEART, ControlNetType.HED, ControlNetType.MLSD]:
            processor = self.processors[controlnet.type.value]
            processed_img = processor(img, output_type='cv2')
        else:
            processed_img = img

        # Resize the image
        height, width, _ = processed_img.shape
        ratio = np.sqrt(1024. * 1024. / (width * height))
        new_width, new_height = int(width * ratio), int(height * ratio)
        processed_img = cv2.resize(processed_img, (new_width, new_height))
        
        return Image.fromarray(processed_img)

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Generate an image based on the input prompt and ControlNets."""
        if not self.pipeline:
            raise RuntimeError("Model not initialized. Call setup() first.")

        # If a custom model URL is provided, load it
        if input_data.model_url != self.default_model_url:
            eulera_scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
                input_data.model_url, subfolder="scheduler"
            )
            vae = AutoencoderKL.from_pretrained(
                "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
            )
            controlnet_model = ControlNetModel_Union.from_pretrained(
                "xinsir/controlnet-union-sdxl-1.0",
                torch_dtype=torch.float16,
                use_safetensors=True
            )

            self.pipeline = StableDiffusionXLControlNetUnionPipeline.from_pretrained(
                input_data.model_url,
                controlnet=controlnet_model,
                vae=vae,
                torch_dtype=torch.float16,
                scheduler=eulera_scheduler,
            )
            self.pipeline.to("cuda")

        # Prepare ControlNet inputs
        image_list = [0] * 6  # Initialize with zeros for all 6 possible ControlNet types
        control_type = torch.zeros(6)  # Initialize control type tensor

        # Map ControlNet types to their indices
        type_to_index = {
            ControlNetType.OPENPOSE: 0,
            ControlNetType.DEPTH: 1,
            ControlNetType.HED: 2,
            ControlNetType.CANNY: 3,
            ControlNetType.LINEART: 3,
            ControlNetType.ANIME_LINEART: 3,
            ControlNetType.MLSD: 3,
            ControlNetType.NORMAL: 4,
            ControlNetType.SEGMENT: 5
        }

        # Process each ControlNet
        for controlnet in input_data.controlnets:
            idx = type_to_index[controlnet.type]
            processed_image = self._process_controlnet_image(controlnet)
            image_list[idx] = processed_image
            control_type[idx] = controlnet.strength

        # Generate the image
        generator = torch.Generator('cuda').manual_seed(torch.randint(0, 2147483647, (1,)).item())
        
        images = self.pipeline(
            prompt=[input_data.prompt],
            image_list=image_list,
            negative_prompt=[input_data.negative_prompt] if input_data.negative_prompt else None,
            generator=generator,
            width=input_data.width,
            height=input_data.height,
            num_inference_steps=input_data.num_inference_steps,
            guidance_scale=input_data.guidance_scale,
            union_control=True,
            union_control_type=control_type,
        ).images

        # Save the image
        output_path = "/tmp/generated_image.png"
        images[0].save(output_path)

        return AppOutput(result=File(path=output_path))


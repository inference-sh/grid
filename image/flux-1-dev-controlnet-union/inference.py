import os
import sys

# Add ControlNetPlus folder to Python path
controlnet_plus_path = os.path.join(os.path.dirname(__file__), "ControlNetPlus")
sys.path.append(controlnet_plus_path)

from typing import Optional, List
from pydantic import Field
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from diffusers import FluxControlNetPipeline, FluxControlNetModel
from controlnet_aux import ZoeDetector
import torch
import cv2
import numpy as np
from PIL import Image
from enum import Enum

class ControlNetType(str, Enum):
    CANNY = "canny"  # 0
    TILE = "tile"    # 1
    DEPTH = "depth"  # 2
    BLUR = "blur"    # 3
    POSE = "pose"    # 4
    GRAY = "gray"    # 5
    LOW_QUALITY = "low_quality"  # 6

    def to_int(self) -> int:
        """Convert ControlNetType to its corresponding integer value."""
        mapping = {
            ControlNetType.CANNY: 0,
            ControlNetType.TILE: 1,
            ControlNetType.DEPTH: 2,
            ControlNetType.BLUR: 3,
            ControlNetType.POSE: 4,
            ControlNetType.GRAY: 5,
            ControlNetType.LOW_QUALITY: 6
        }
        return mapping[self]

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

class AppInput(BaseAppInput):
    prompt: str = Field(
        ...,
        description="The text prompt to generate the image from",
        examples=["A majestic lion jumping from a big stone at night"]
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
    controlnet_type: ControlNetType = Field(
        None,
        description="Type of ControlNet to use"
    )
    controlnet_image: File = Field(
        None,
        description="Input image for the ControlNet"
    )
    controlnet_strength: float = Field(
        1.0,
        description="Strength of the ControlNet effect",
        ge=0.0,
        le=1.0
    )
    controlnet_pre_process: bool = Field(
        True,
        description="Whether to pre-process the input image",
    )
    control_guidance_start: float = Field(
        0.0,
        description="When to start applying ControlNet guidance (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    control_guidance_end: float = Field(
        1.0,
        description="When to stop applying ControlNet guidance (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )

class AppOutput(BaseAppOutput):
    result: File

class App(BaseApp):
    pipeline: Optional[FluxControlNetPipeline] = None
    processor: Optional[ZoeDetector] = None

    async def setup(self, metadata):
        """Initialize the FLUX model with ControlNet support."""
        # Initialize depth processor
        self.processor = ZoeDetector.from_pretrained("lllyasviel/Annotators").to("cuda")

        # Initialize pipeline components
        controlnet_model = FluxControlNetModel.from_pretrained(
            "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro",
            torch_dtype=torch.float16,
            use_safetensors=True
        )

        self.pipeline = FluxControlNetPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            controlnet=controlnet_model,
            torch_dtype=torch.float16,
        )
        self.pipeline.to("cuda")

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Generate an image based on the input prompt and ControlNet."""
        if not self.pipeline:
            raise RuntimeError("Model not initialized. Call setup() first.")

        # Process ControlNet input if provided
        control_image = None
        control_mode = None
        control_scale = None

        if input_data.controlnet_type and input_data.controlnet_image:
            # Read the image
            img = cv2.imread(input_data.controlnet_image.path)
            
            # If pre_process is False, just resize the image and return
            if not input_data.controlnet_pre_process:
                height, width, _ = img.shape
                ratio = np.sqrt(1024. * 1024. / (width * height))
                new_width, new_height = int(width * ratio), int(height * ratio)
                processed_img = cv2.resize(img, (new_width, new_height))
                control_image = Image.fromarray(processed_img)
            else:
                # Process based on ControlNet type
                if input_data.controlnet_type == ControlNetType.CANNY:
                    processed_img = cv2.Canny(img, 100, 200)
                    processed_img = HWC3(processed_img)
                elif input_data.controlnet_type == ControlNetType.DEPTH:
                    processed_img = self.processor(img, output_type='cv2')
                elif input_data.controlnet_type == ControlNetType.POSE:
                    processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    processed_img = HWC3(processed_img)
                elif input_data.controlnet_type == ControlNetType.GRAY:
                    processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    processed_img = HWC3(processed_img)
                elif input_data.controlnet_type == ControlNetType.BLUR:
                    processed_img = cv2.GaussianBlur(img, (5, 5), 0)
                elif input_data.controlnet_type == ControlNetType.TILE:
                    processed_img = img
                elif input_data.controlnet_type == ControlNetType.LOW_QUALITY:
                    height, width = img.shape[:2]
                    processed_img = cv2.resize(img, (width//4, height//4))
                    processed_img = cv2.resize(processed_img, (width, height))
                else:
                    processed_img = img

                # Resize the image
                height, width, _ = processed_img.shape
                ratio = np.sqrt(1024. * 1024. / (width * height))
                new_width, new_height = int(width * ratio), int(height * ratio)
                processed_img = cv2.resize(processed_img, (new_width, new_height))
                
                control_image = Image.fromarray(processed_img)

            control_mode = input_data.controlnet_type.to_int()
            control_scale = input_data.controlnet_strength

        # Generate the image
        generator = torch.Generator('cuda').manual_seed(torch.randint(0, 2147483647, (1,)).item())
        
        print("Control Image: ", control_image)
        print("Mode: ", control_mode)
        print("Scale: ", control_scale)

        images = self.pipeline(
            prompt=input_data.prompt,
            control_image=control_image,
            control_mode=control_mode,
            generator=generator,
            width=input_data.width,
            height=input_data.height,
            num_inference_steps=input_data.num_inference_steps,
            guidance_scale=input_data.guidance_scale,
            control_guidance_start=input_data.control_guidance_start,
            control_guidance_end=input_data.control_guidance_end,
            controlnet_conditioning_scale=control_scale,
        ).images

        # Save the image
        output_path = "/tmp/generated_image.png"
        images[0].save(output_path)

        return AppOutput(result=File(path=output_path))

    async def unload(self):
        """Clean up resources."""
        if self.pipeline:
            self.pipeline = None
        if self.processor:
            self.processor.to("cpu")
        self.processor = None
        torch.cuda.empty_cache()
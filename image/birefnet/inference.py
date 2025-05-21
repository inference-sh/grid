from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from PIL import Image
import torch
from transformers import AutoModelForImageSegmentation
import numpy as np
import os
from typing import Optional
from pydantic import Field
from torchvision import transforms

class AppInput(BaseAppInput):
    image: File = Field(
        ...,
        description="Input image file for background removal",
        example="path/to/image.jpg"
    )
    return_mask: bool = Field(
        False,
        description="Whether to use the matting model to return a mask (True) or the normal model for binary mask (False)",
        example=False
    )

class AppOutput(BaseAppOutput):
    result: File = Field(
        ...,
        description="Output image file with transparent background in PNG format",
        example="path/to/transparent.png"
    )

class App(BaseApp):
    matting_model: Optional[AutoModelForImageSegmentation] = Field(
        None,
        description="BiRefNet model for image matting with transparency"
    )
    normal_model: Optional[AutoModelForImageSegmentation] = Field(
        None,
        description="BiRefNet model for binary mask generation"
    )
    device: str = Field(
        "cuda" if torch.cuda.is_available() else "cpu",
        description="Device to run the model on (cuda or cpu)"
    )

    async def setup(self, metadata):
        """Initialize both BiRefNet models."""
        # Load matting model
        self.matting_model = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet_HR-matting",
            trust_remote_code=True
        ).to(self.device)
        self.matting_model.eval()
        self.matting_model.half()

        # Load normal model
        self.normal_model = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet_HR",
            trust_remote_code=True
        ).to(self.device)
        self.normal_model.eval()
        self.normal_model.half()

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Run background removal on the input image."""
        # Load and preprocess image
        image = Image.open(input_data.image.path).convert("RGB")
        
        # Resize image to 2048x2048 as recommended by the model
        image_size = (2048, 2048)
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Prepare input tensor
        input_tensor = transform(image).unsqueeze(0).to(self.device).half()
        
        # Select model based on input
        model = self.matting_model if input_data.return_mask else self.normal_model
        
        # Run inference
        with torch.no_grad():
            pred = model(input_tensor)[-1].sigmoid().cpu()
        
        # Convert prediction to mask
        mask = pred[0].squeeze()
        mask = (mask * 255).byte().numpy()
        mask = Image.fromarray(mask).resize(image.size)
        
        if not input_data.return_mask:
            # Create transparent image
            transparent_image = image.copy()
            transparent_image.putalpha(mask)
            output_image = transparent_image
        else:
            # Create binary mask image
            output_image = mask
        
        # Save result
        output_path = "/tmp/result.png"
        output_image.save(output_path, "PNG")
        
        return AppOutput(result=File(path=output_path))

    async def unload(self):
        """Clean up resources."""
        if self.matting_model is not None:
            del self.matting_model
        if self.normal_model is not None:
            del self.normal_model
        torch.cuda.empty_cache()
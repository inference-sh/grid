from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from PIL import Image
import torch
from transformers import AutoModelForImageSegmentation
from typing import Optional
from pydantic import Field
from pydantic.color import Color
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
    fill_background: bool = Field(
        False,
        description="Fill the background with a solid color instead of returning transparency",
        example=False
    )
    background_color: Color = Field(
        "#FFFFFFFF",
        description="Background color to use when fill_background is True",
        example="#FFFFFFFF"
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
        mask_image = Image.fromarray(mask).resize(image.size)
        
        if input_data.return_mask:
            # Create binary mask image
            output_image = mask_image
        elif input_data.fill_background:
            background_rgb = tuple(int(channel) for channel in input_data.background_color.as_rgb_tuple())
            background = Image.new("RGB", image.size, background_rgb)
            output_image = Image.composite(image, background, mask_image)
        else:
            # Create transparent image
            transparent_image = image.copy()
            transparent_image.putalpha(mask_image)
            output_image = transparent_image
        
        # Save result
        output_path = "/tmp/result.png"
        output_image.save(output_path, "PNG")
        
        return AppOutput(result=File(path=output_path))

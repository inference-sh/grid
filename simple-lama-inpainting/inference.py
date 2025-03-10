from .simple_lama import SimpleLama
from PIL import Image
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field, BaseModel
from typing import List
from io import BytesIO
from PIL import Image
import base64

def image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

class ImageMask(BaseModel):
    image: File = Field(
        description="The original image you want to fix or modify.",
        example=File(
            path="https://example.com/sample-image.jpg",
            mime_type="image/jpeg",
            filename="sample-image.jpg"
        ),
        default=File(
            path="/path/to/default-image.jpg",
            mime_type="image/jpeg",
            filename="default-image.jpg"
        )
    )
    mask: File = Field(
        description="The mask image that shows which areas to inpaint (white = replace, black = keep). Must be the same size as the input image.",
        example=File(
            path="https://example.com/sample-mask.png",
            mime_type="image/png",
            filename="sample-mask.png"
        ),
        default=File(
            path="/path/to/default-mask.png",
            mime_type="image/png",
            filename="default-mask.png"
        )
    )

class AppInput(BaseAppInput):
    inputs: List[ImageMask] = Field(description="The images to inpaint with. Must be the same size as the image.")

class AppOutput(BaseAppOutput):
    image: File = Field(description="The inpainted image")

class App(BaseApp):
    lama: SimpleLama | None = None
    async def setup(self):
        self.lama = SimpleLama()

    async def run(self, app_input: AppInput) -> AppOutput:

        image = Image.open(app_input.image.path).convert("RGB")
        mask = Image.open(app_input.mask.path).convert("L")

        result = self.lama(image, mask)
        result_path = "/tmp/result.png"
        result.save(result_path)
        output = AppOutput(image=File(path=result_path))
        return output

    async def unload(self):
        self.lama = None
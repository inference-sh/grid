from .simple_lama import SimpleLama
from PIL import Image
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field, BaseModel

class AppInput(BaseAppInput):
    image: File = Field(
        description="The original image you want to fix or modify.",
        examples=[
            File(
                uri="https://1nf.sh/samples/bike.png"
            ),
        ],
        default=File(
            uri="https://1nf.sh/samples/bike.png"
        )
    )
    mask: File = Field(
        description="The mask image that shows which areas to inpaint (white = replace, black = keep). Must be the same size as the input image.",
        examples=[
            File(
                uri="https://1nf.sh/samples/bike_mask.png"
            ),
        ],
        default=File(
            uri="https://1nf.sh/samples/bike_mask.png"
        )
    )

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
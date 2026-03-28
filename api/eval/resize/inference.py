import logging
from typing import Literal
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta
from pydantic import Field
from PIL import Image


class RunInput(BaseAppInput):
    image: File = Field(description="Image to crop/resize")
    size: int = Field(default=1024, description="Output size in pixels (square)")
    mode: Literal["center", "cover"] = Field(default="center", description="center: crop from center to square then resize. cover: resize shortest side to target, crop center")


class RunOutput(BaseAppOutput):
    image: File = Field(description="Cropped and resized image")


class App(BaseApp):
    async def setup(self, config):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Resize app ready")

    async def run(self, input_data: RunInput) -> RunOutput:
        img = Image.open(input_data.image.path).convert("RGB")
        w, h = img.size
        target = input_data.size

        print(f"[resize] Input: {w}x{h}, target: {target}x{target}, mode: {input_data.mode}")

        if input_data.mode == "center":
            # Center crop to square first
            side = min(w, h)
            left = (w - side) // 2
            top = (h - side) // 2
            img = img.crop((left, top, left + side, top + side))
        else:
            # Cover: scale shortest side to target, then center crop
            scale = target / min(w, h)
            img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
            w, h = img.size
            left = (w - target) // 2
            top = (h - target) // 2
            img = img.crop((left, top, left + target, top + target))

        # Resize to exact target
        img = img.resize((target, target), Image.LANCZOS)

        output_path = "/tmp/resized.png"
        img.save(output_path, "PNG")

        print(f"[resize] Output: {target}x{target}")

        return RunOutput(
            image=File(path=output_path),
            output_meta=OutputMeta(
                outputs=[ImageMeta(width=target, height=target, count=1)]
            ),
        )

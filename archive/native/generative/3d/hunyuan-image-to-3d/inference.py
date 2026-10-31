import os
import sys
import subprocess
import torch
from PIL import Image
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
class AppInput(BaseAppInput):
    prompt: str = Field(
        default="",
        description="Optional text prompt to guide the 3D generation",
        examples=["a red sports car", "a wooden chair"],
        max_length=1000
    )

    input_image: File = Field(
        description="Input image to convert to 3D model",
        examples=["https://1nf.sh/examples/car.jpg", "https://1nf.sh/examples/chair.png"]
    )

    num_inference_steps: int = Field(
        default=30,
        description="Number of denoising steps (higher = better quality but slower)",
        ge=1,
        le=100,
        examples=[30, 50]
    )

    seed: int = Field(
        default=2025,
        description="Random seed for reproducible results",
        ge=0,
        examples=[2025, 42]
    )

    paint_texture: bool = Field(
        default=False,
        description="Whether to paint texture on the 3D model"
    )

    class Config:
        json_schema_extra = {
            "title": "Image to 3D Model Generation Input",
            "description": "Input parameters for converting an image to a 3D model",
            "examples": [{
                "prompt": "a red sports car",
                "input_image": {"filename": "car.jpg"},
                "num_inference_steps": 30,
                "seed": 2025,
                "paint_texture": False
            }]
        }

class AppOutput(BaseAppOutput):
    result: File = Field(
        description="Generated 3D model file"
    )

class App(BaseApp):
    async def setup(self, metadata):
        """Initialize Hunyuan3D model and dependencies"""
        original_dir = os.getcwd()
        
        try:
            if original_dir not in sys.path:
                sys.path.insert(0, original_dir)
            
            # Install custom rasterizer
            print("Installing custom rasterizer...")
            current_file_path = os.path.dirname(os.path.abspath(__file__))
            os.chdir(os.path.join(current_file_path, "hy3dgen/texgen/custom_rasterizer"))
            subprocess.run(["python3", "setup.py", "build_ext", "--inplace"], check=True)
            subprocess.run(["python3", "setup.py", "install"], check=True)
            
            custom_rasterizer_dir = os.getcwd()
            if custom_rasterizer_dir not in sys.path:
                sys.path.insert(0, custom_rasterizer_dir)
            
            os.chdir("../../..")
            
            # Install differentiable renderer
            print("Installing differentiable renderer...")
            os.chdir(current_file_path)
            os.chdir(os.path.join(current_file_path, "hy3dgen/texgen/differentiable_renderer"))
            subprocess.run(["python3", "setup.py", "install"], check=True)
            
        finally:
            print("Finished installing dependencies")
            os.chdir(original_dir)
            
        sys.path.append(current_file_path)

        # Import dependencies
        from hy3dgen.rembg import BackgroundRemover
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FaceReducer, FloaterRemover, DegenerateFaceRemover
        from hy3dgen.text2image import HunyuanDiTPipeline
        from hy3dgen.texgen import Hunyuan3DPaintPipeline

        self.device = "cuda"
        self.model_path = 'tencent/Hunyuan3D-2'
        
        # Initialize components
        self.rembg = BackgroundRemover()
        self.i23d_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(self.model_path, device=self.device)
        self.t2i_pipeline = HunyuanDiTPipeline('Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled', device=self.device)
        self.floater_remover = FloaterRemover()
        self.face_remover = DegenerateFaceRemover()
        self.face_reducer = FaceReducer()
        self.paint_pipeline = Hunyuan3DPaintPipeline.from_pretrained(self.model_path)

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Generate 3D model from image or text"""
        print(f"Running Hunyuan3D model with input data: {input}")
        if input_data.input_image and not input_data.prompt:
            # Image-to-3D path
            image = Image.open(input_data.input_image.path)
            if image.mode == 'RGB':
                image = self.rembg(image)
        elif input_data.prompt and not input_data.input_image:
            # Text-to-3D path
            image = self.t2i_pipeline(input_data.prompt)
            image = self.rembg(image)
        else:
            raise ValueError("Please provide either an input image OR a text prompt, not both or neither")

        # Generate 3D mesh
        mesh = self.i23d_pipeline(
            image=image,
            num_inference_steps=input_data.num_inference_steps,
            mc_algo='mc',
            generator=torch.manual_seed(input_data.seed)
        )[0]

        # Post-process mesh
        mesh = self.floater_remover(mesh)
        mesh = self.face_remover(mesh)
        mesh = self.face_reducer(mesh)

        # Paint texture if enabled
        if input_data.paint_texture:
            mesh = self.paint_pipeline(mesh, image=image)

        # Save and return result
        output_path = "/tmp/output.glb"
        mesh.export(output_path)
        return AppOutput(result=File.from_path(output_path))
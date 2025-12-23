import os
import sys
import subprocess
import torch
from PIL import Image
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
import logging
from typing import List

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

class AppInput(BaseAppInput):
    prompt: str = Field(
        default="",
        description="Optional text prompt to guide the 3D generation",
        examples=["a red sports car", "a wooden chair"],
        max_length=1000
    )

    input_image: File = Field(
        default=None,
        description="Input image to convert to 3D model (optional if prompt is provided)",
        examples=["https://1nf.sh/examples/car.jpg", "https://1nf.sh/examples/chair.png"]
    )

    additional_images: List[File] = Field(
        default_factory=list,
        description="Additional images for multiview input (used for mv/mv_turbo variants)",
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

    background_removal: bool = Field(
        default=True,
        description="Whether to apply background removal to input images"
    )

    floater_remover: bool = Field(
        default=True,
        description="Whether to apply floater removal post-processing"
    )

    face_remover: bool = Field(
        default=True,
        description="Whether to apply degenerate face removal post-processing"
    )

    face_reducer: bool = Field(
        default=True,
        description="Whether to apply face reduction post-processing"
    )

    paint_texture: bool = Field(
        default=True,
        description="Whether to paint texture on the 3D model"
    )

    class Config:
        json_schema_extra = {
            "title": "Image/Text to 3D Model Generation Input",
            "description": "Input parameters for converting an image or text to a 3D model using Hunyuan3D-2",
            "examples": [{
                "prompt": "a red sports car",
                "input_image": None,
                "additional_images": [],
                "num_inference_steps": 30,
                "seed": 2025,
                "background_removal": True,
                "floater_remover": True,
                "face_remover": True,
                "face_reducer": True,
                "paint_texture": True
            }]
        }

class AppOutput(BaseAppOutput):
    result: File = Field(
        description="Generated 3D model file in GLB format"
    )

# Model variant configurations for Hunyuan3D-2
MODEL_VARIANTS = {
    "mini": {
        "repo_id": "tencent/Hunyuan3D-2mini",
        "subfolder": "hunyuan3d-dit-v2-mini",
        "turbo_subfolder": "hunyuan3d-dit-v2-mini-turbo",
        "texgen_repo": "tencent/Hunyuan3D-2",
        "texgen_subfolder": "hunyuan3d-paint-v2-0"
    },
    "mini_turbo": {
        "repo_id": "tencent/Hunyuan3D-2mini",
        "subfolder": "hunyuan3d-dit-v2-mini-turbo",
        "turbo_subfolder": "hunyuan3d-dit-v2-mini-turbo",
        "texgen_repo": "tencent/Hunyuan3D-2",
        "texgen_subfolder": "hunyuan3d-paint-v2-0-turbo",
        "enable_flashvdm": True
    },
    "mv": {
        "repo_id": "tencent/Hunyuan3D-2mv",
        "subfolder": "hunyuan3d-dit-v2-mv",
        "turbo_subfolder": "hunyuan3d-dit-v2-mv-turbo",
        "texgen_repo": "tencent/Hunyuan3D-2",
        "texgen_subfolder": "hunyuan3d-paint-v2-0"
    },
    "mv_turbo": {
        "repo_id": "tencent/Hunyuan3D-2mv",
        "subfolder": "hunyuan3d-dit-v2-mv-turbo",
        "turbo_subfolder": "hunyuan3d-dit-v2-mv-turbo",
        "texgen_repo": "tencent/Hunyuan3D-2",
        "texgen_subfolder": "hunyuan3d-paint-v2-0-turbo",
        "enable_flashvdm": True
    },
    "default": {
        "repo_id": "tencent/Hunyuan3D-2",
        "subfolder": "hunyuan3d-dit-v2-0",
        "turbo_subfolder": "hunyuan3d-dit-v2-0-turbo",
        "texgen_repo": "tencent/Hunyuan3D-2",
        "texgen_subfolder": "hunyuan3d-paint-v2-0"
    },
    "standard_turbo": {
        "repo_id": "tencent/Hunyuan3D-2",
        "subfolder": "hunyuan3d-dit-v2-0-turbo",
        "turbo_subfolder": "hunyuan3d-dit-v2-0-turbo",
        "texgen_repo": "tencent/Hunyuan3D-2",
        "texgen_subfolder": "hunyuan3d-paint-v2-0-turbo",
        "enable_flashvdm": True
    }
}

DEFAULT_VARIANT = "mini"

class App(BaseApp):
    async def setup(self, metadata):
        """Initialize Hunyuan3D-2 model and dependencies"""
        logging.basicConfig(level=logging.INFO)
        
        # Get variant from metadata
        variant = getattr(metadata, "app_variant", DEFAULT_VARIANT)
        if variant not in MODEL_VARIANTS:
            logging.warning(f"Unknown variant '{variant}', falling back to default '{DEFAULT_VARIANT}'")
            variant = DEFAULT_VARIANT
        
        self.variant_config = MODEL_VARIANTS[variant]
        logging.info(f"Using variant: {variant}")
        
        original_dir = os.getcwd()
        
        try:
            if original_dir not in sys.path:
                sys.path.insert(0, original_dir)
            
            # Install custom rasterizer
            logging.info("Installing custom rasterizer...")
            current_file_path = os.path.dirname(os.path.abspath(__file__))
            os.chdir(os.path.join(current_file_path, "hy3dgen/texgen/custom_rasterizer"))
            subprocess.run(["python3", "setup.py", "build_ext", "--inplace"], check=True)
            subprocess.run(["python3", "setup.py", "install"], check=True)
            
            custom_rasterizer_dir = os.getcwd()
            if custom_rasterizer_dir not in sys.path:
                sys.path.insert(0, custom_rasterizer_dir)
            
            os.chdir("../../..")
            
            # Install differentiable renderer
            logging.info("Installing differentiable renderer...")
            os.chdir(current_file_path)
            os.chdir(os.path.join(current_file_path, "hy3dgen/texgen/differentiable_renderer"))
            subprocess.run(["python3", "setup.py", "install"], check=True)
            
        finally:
            logging.info("Finished installing dependencies")
            os.chdir(original_dir)
            
        sys.path.append(current_file_path)

        # Import dependencies
        from hy3dgen.rembg import BackgroundRemover
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FaceReducer, FloaterRemover, DegenerateFaceRemover
        from hy3dgen.text2image import HunyuanDiTPipeline
        from hy3dgen.texgen import Hunyuan3DPaintPipeline

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {self.device}")
        
        # Initialize components
        self.rembg = BackgroundRemover()
        
        # Load shape generation pipeline with variant-specific configuration
        logging.info(f"Loading shape generation model from {self.variant_config['repo_id']}...")
        self.i23d_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            self.variant_config['repo_id'], 
            subfolder=self.variant_config['subfolder'],
            device=self.device
        )
        
        # Load text-to-image pipeline
        logging.info("Loading text-to-image model...")
        self.t2i_pipeline = HunyuanDiTPipeline(
            'Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled', 
            device=self.device
        )
        
        # Load post-processing components
        self.floater_remover = FloaterRemover()
        self.face_remover = DegenerateFaceRemover()
        self.face_reducer = FaceReducer()
        
        # Load texture painting pipeline
        logging.info(f"Loading texture painting model from {self.variant_config['texgen_repo']}...")
        self.paint_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
            self.variant_config['texgen_repo'],
            subfolder=self.variant_config['texgen_subfolder']
        )

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Generate 3D model from image or text"""
        logging.info(f"Running Hunyuan3D-2 model with variant: {getattr(metadata, 'app_variant', DEFAULT_VARIANT)}")
        
        # Validate input
        if not input_data.input_image and not input_data.prompt:
            raise ValueError("Please provide either an input image OR a text prompt")
        
        if input_data.input_image and input_data.prompt:
            logging.warning("Both image and prompt provided, prioritizing image input")
        
        # Prepare image(s) for pipeline
        variant = getattr(metadata, "app_variant", DEFAULT_VARIANT)
        is_multiview = variant in ["mv", "mv_turbo"]
        
        if is_multiview and input_data.additional_images:
            # Multiview expects a dict of view_name: image
            view_names = ["front", "left", "back", "right", "top", "bottom"]
            images = {}
            for idx, file in enumerate(input_data.additional_images):
                if idx < len(view_names):
                    img = Image.open(file.path)
                    if input_data.background_removal and img.mode == 'RGB':
                        img = self.rembg(img)
                    images[view_names[idx]] = img
            if not images:
                raise ValueError("No valid additional images provided for multiview variant.")
        else:
            # Single image path
            if input_data.input_image:
                image = Image.open(input_data.input_image.path)
                if input_data.background_removal and image.mode == 'RGB':
                    image = self.rembg(image)
            else:
                image = self.t2i_pipeline(input_data.prompt)
                if input_data.background_removal:
                    image = self.rembg(image)
        
        # Generate 3D mesh
        logging.info("Generating 3D mesh...")
        generator = torch.manual_seed(input_data.seed) if input_data.seed else None
        use_flashvdm = self.variant_config.get('enable_flashvdm', False)
        
        pipeline_input = images if is_multiview and input_data.additional_images else image
        mesh = self.i23d_pipeline(
            image=pipeline_input,
            num_inference_steps=input_data.num_inference_steps,
            mc_algo='mc',
            generator=generator,
            enable_flashvdm=use_flashvdm
        )[0]

        # Post-process mesh
        logging.info("Post-processing mesh...")
        if input_data.floater_remover:
            mesh = self.floater_remover(mesh)
        if input_data.face_remover:
            mesh = self.face_remover(mesh)
        if input_data.face_reducer:
            mesh = self.face_reducer(mesh)

        # Paint texture if enabled
        if input_data.paint_texture:
            logging.info("Painting texture on mesh...")
            mesh = self.paint_pipeline(mesh, image=image if not is_multiview else list(images.values())[0])

        # Save and return result
        output_path = "/tmp/output.glb"
        logging.info(f"Saving result to {output_path}")
        mesh.export(output_path)
        return AppOutput(result=File.from_path(output_path))
import os
import sys
import json
import tempfile
from typing import Optional
from dataclasses import dataclass

import torch
from accelerate import Accelerator
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf
from pydantic import Field

# Add parent directory to Python path so we can import common module
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File

# Import Humo specific modules
from .generate import Generator

@dataclass
class ModelConfig:
    mode: str = "TIA"  # TIA or TA
    frames: int = 97
    scale_a: float = 5.5
    scale_t: float = 5.0
    height: int = 720
    width: int = 1280
    sampling_steps: int = 50
    
class AppInput(BaseAppInput):
    prompt: str = Field(description="Text prompt describing the video to generate")
    frames: int = Field(description="Number of frames to generate", default=97)
    height: int = Field(description="Output video height", default=720)
    width: int = Field(description="Output video width", default=1280)
    scale_a: float = Field(description="Audio guidance scale", default=5.5)
    scale_t: float = Field(description="Text guidance scale", default=5.0)
    sampling_steps: int = Field(description="Number of diffusion sampling steps", default=50)
    image: Optional[File] = Field(None, description="Optional reference image - if provided, uses TIA mode instead of TA mode")
    audio: File = Field(description="Reference audio file for generation")

class AppOutput(BaseAppOutput):
    video: File = Field(description="Generated video file")
    frames_dir: Optional[str] = Field(None, description="Directory containing individual frames (if requested)")

class App(BaseApp):
    def __init__(self):
        super().__init__()
        self.model = None
        self.accelerator = None
        self.device = None
        self.config = None
        self.model_path = None
        
    async def setup(self, metadata):
        """Initialize models and resources."""
        # Initialize accelerator for device management
        self.accelerator = Accelerator()
        self.device = self.accelerator.device

        # Download complete model repositories from HuggingFace
        wan_dir = snapshot_download(repo_id="Wan-AI/Wan2.1-T2V-1.3B")
        humo_dir = snapshot_download(repo_id="bytedance-research/HuMo")
        whisper_dir = snapshot_download(repo_id="openai/whisper-large-v3")
        vocal_separator_dir = snapshot_download(repo_id="huangjackson/Kim_Vocal_2")

        # Get variant configuration
        variant = getattr(metadata, "app_variant", "default")
        
        # Configure based on variant
        if variant == "default":
            self.config = ModelConfig()
        elif variant == "high_quality":
            self.config = ModelConfig(
                sampling_steps=75,
                height=1080,
                width=1920
            )
        elif variant == "low_vram":
            self.config = ModelConfig(
                height=480,
                width=854,
                sampling_steps=30
            )
        else:
            self.config = ModelConfig()

        # Create temporary prompt file
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = "configs/inference/generate.yaml"  # Path relative to project root
        
        # Update config with downloaded model paths
        self.model_paths = {
            "vae": os.path.join(wan_dir, "Wan2.1_VAE.pth"),
            "t5_model": os.path.join(wan_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
            "t5_tokenizer": os.path.join(wan_dir, "google/umt5-xxl"),
            "zero_vae": os.path.join(humo_dir, "zero_vae_129frame.pt"),
            "zero_vae_720p": os.path.join(humo_dir, "zero_vae_720p_161frame.pt"),
            "humo_model": os.path.join(humo_dir, "HuMo-17B"),
            "whisper": whisper_dir,
            "vocal_separator": os.path.join(vocal_separator_dir, "Kim_Vocal_2.onnx")
        }

        # Load configs
        self.base_config = OmegaConf.load(self.config_path)
        wan_config = OmegaConf.load("configs/models/Wan_14B_I2V.yaml")
        
        # Update config with downloaded model paths
        self.base_config.dit.zero_vae_path = self.model_paths["zero_vae"]
        self.base_config.dit.zero_vae_720p_path = self.model_paths["zero_vae_720p"]
        self.base_config.dit.checkpoint_dir = self.model_paths["humo_model"]
        self.base_config.vae.checkpoint = self.model_paths["vae"]
        self.base_config.text.t5_checkpoint = self.model_paths["t5_model"]
        self.base_config.text.t5_tokenizer = self.model_paths["t5_tokenizer"]
        self.base_config.audio.vocal_separator = self.model_paths["vocal_separator"]
        self.base_config.audio.wav2vec_model = self.model_paths["whisper"]
        
        # Just merge Wan config into model section and set required fields
        self.base_config.dit.model = wan_config
        # self.base_config.dit.model.model_type = "i2v"
        # self.base_config.dit.model.insert_audio = True
        
        # Create model with config
        self.model = Generator(self.base_config)
        self.model.entrypoint()

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Run video generation."""
        # Create temporary prompt file
        prompt_file = os.path.join(self.temp_dir, "prompt.json")
        with open(prompt_file, "w") as f:
            json.dump({"prompt": input_data.prompt}, f)

        # Create output directory
        output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        # Determine mode based on whether image is provided
        mode = "TIA" if input_data.image is not None else "TA"

        # Prepare generation parameters
        generation_args = [
            f"generation.frames={input_data.frames}",
            f"generation.scale_a={input_data.scale_a}",
            f"generation.scale_t={input_data.scale_t}",
            f"generation.mode={mode}",
            f"generation.height={input_data.height}",
            f"generation.width={input_data.width}",
            f"diffusion.timesteps.sampling.steps={input_data.sampling_steps}",
            f"generation.positive_prompt={prompt_file}",
            f"generation.output.dir={output_dir}"
        ]

        # Add reference files
        if input_data.image is not None:
            generation_args.append(f"generation.reference_image={input_data.image.path}")
        generation_args.append(f"generation.reference_audio={input_data.audio.path}")

        # Update config with generation args
        config = OmegaConf.merge(self.base_config, OmegaConf.from_dotlist(generation_args))
        
        # Generate video
        self.model.generate(config)

        # Find generated video file
        video_file = None
        for file in os.listdir(output_dir):
            if file.endswith(".mp4"):
                video_file = os.path.join(output_dir, file)
                break

        if not video_file:
            raise RuntimeError("Video generation failed - no output file found")

        return AppOutput(
            video=File(path=video_file),
            frames_dir=output_dir if os.path.exists(os.path.join(output_dir, "frames")) else None
        )


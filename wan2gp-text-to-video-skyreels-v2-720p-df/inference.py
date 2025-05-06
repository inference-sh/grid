import os
import sys
from pathlib import Path

# Force profile to 1 by adding it to sys.argv before any imports
if "--profile" not in sys.argv:
    sys.argv.append("--profile")
    sys.argv.append("1")

current_dir = Path(__file__).parent.absolute()
sys.path.append(os.path.join(str(current_dir), "wan"))
sys.path.append(os.path.join(str(current_dir), "wan", "wan"))

import torch
from typing import Optional
from pydantic import Field
import tempfile
import shutil
import subprocess
import numpy as np

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from .wan.wan.configs import WAN_CONFIGS
from mmgp import offload

# Import core functions from wgp
from .wan.wgp import (
    download_models,
    generate_video
)

def send_cmd(cmd, message="", *args, **kwargs):
    if cmd != "preview":
        print(f"{cmd}: {message}")

class AppInput(BaseAppInput):
    prompt: str = Field(description="Text prompt for video generation")
    size: str = Field(default="832x480", description="Size of the generated video (width*height)")
    num_frames: int = Field(default=81, description="Number of frames to generate (should be 4n+1)")
    fps: int = Field(default=16, description="Frames per second for the output video")
    guidance_scale: float = Field(default=5.0, description="Classifier-free guidance scale")
    num_inference_steps: int = Field(default=30, description="Number of denoising steps")
    seed: Optional[int] = Field(default=-1, description="Random seed for reproducibility (-1 for random)")
    negative_prompt: str = Field(default="", description="Negative prompt to guide generation")
    sample_solver: str = Field(default="unipc", description="Solver to use for sampling (unipc or dpm++)")
    shift: float = Field(default=5.0, description="Noise schedule shift parameter")
    tea_cache: float = Field(default=2.0, description="TeaCache multiplier (0 to disable, 1.5-2.5 recommended for speed)")
    tea_cache_start_step_perc: int = Field(default=0, description="TeaCache starting step percentage")
    lora_file: Optional[str] = Field(default=None, description="URL to Lora file in safetensors format")
    lora_multiplier: float = Field(default=1.0, description="Multiplier for the Lora effect")
    vae_tile_size: int = Field(default=128, description="VAE tile size for lower VRAM usage (0, 128, or 256)")
    enable_RIFLEx: bool = Field(default=True, description="Enable RIFLEx positional embedding for longer videos")
    joint_pass: bool = Field(default=True, description="Enable joint pass for 10% speed boost")
    attention: str = Field(
        default="sage", 
        description="Attention mechanism to use for generation",
        enum=["sage", "sdpa"]
    )
    cfg_star_switch: bool = Field(default=True, description="Enable CFG* guidance")
    cfg_zero_step: int = Field(default=5, description="Step at which to switch to CFG* guidance")
    temporal_upsampling: str = Field(
        default="",
        description="Temporal upsampling method",
        enum=["", "rife2", "rife4"]
    )
    spatial_upsampling: str = Field(
        default="",
        description="Spatial upsampling method",
        enum=["", "lanczos1.5", "lanczos2"]
    )

class AppOutput(BaseAppOutput):
    video: File = Field(description="Generated video file")

class App(BaseApp):
    async def setup(self):
        """Initialize the app and download required models"""
        # Set device and check capabilities
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_id = 0 if torch.cuda.is_available() else -1
        
        # Check GPU capabilities and set default dtype
        major, minor = torch.cuda.get_device_capability(self.device)
        if major < 8:
            print("Switching to f16 model as GPU architecture doesn't support bf16")
            self.default_dtype = torch.float16
        else:
            self.default_dtype = torch.bfloat16
        
        # Initialize model filenames
        self.t2v_transformer_filename = "sky_reels2_diffusion_forcing_720p_14B_bf16.safetensors"
        self.text_encoder_filename = "models_t5_umt5-xxl-enc-bf16.safetensors"
        self.vae_filename = "Wan2.1_VAE.safetensors"
        
        # Create directories for models and loras
        ckpts_dir = os.path.join(str(current_dir), "ckpts")
        self.ckpts_dir = ckpts_dir
        loras_dir = os.path.join(str(current_dir), "loras")
        self.loras_dir = loras_dir
        
        os.makedirs(ckpts_dir, exist_ok=True)
        os.makedirs(loras_dir, exist_ok=True)
        
        # Print current contents of ckpts directory
        print("\nCurrent contents of ckpts directory:")
        if os.path.exists(ckpts_dir):
            for file in os.listdir(ckpts_dir):
                print(f"  - {file}")
        else:
            print("  Directory does not exist yet")
        
        # Download models
        try:
            print("\nDownloading models...")
            download_models(self.t2v_transformer_filename, self.text_encoder_filename)
            
            # Print contents after download
            print("\nContents of ckpts directory after download:")
            for file in os.listdir(ckpts_dir):
                print(f"  - {file}")
            
            # Verify that all required model files exist
            required_files = [
                os.path.join(ckpts_dir, self.t2v_transformer_filename),
                os.path.join(ckpts_dir, self.text_encoder_filename),
                os.path.join(ckpts_dir, self.vae_filename)
            ]
            
            missing_files = [f for f in required_files if not os.path.exists(f)]
            if missing_files:
                raise FileNotFoundError(f"Missing required model files: {', '.join(missing_files)}")
                
        except Exception as e:
            print(f"Error during model setup: {str(e)}")
            raise
        
        # Set default device if specified
        if self.device_id >= 0:
            torch.set_default_device(self.device)

    async def run(self, input_data: AppInput) -> AppOutput:
        """Run video generation"""
        # Parse size
        width, height = map(int, input_data.size.split('x'))
        size = (width, height)
        
        # Create state dictionary with all necessary keys
        state = {
            "gen": {
                "file_list": [],
                "prompt_no": 0,
                "refresh": 0,
                "progress_status": "Generating video",
                "progress_phase": ("", 0),
                "num_inference_steps": input_data.num_inference_steps,
                "abort": False,
                "selected": 0,
                "last_selected": False,
                "in_progress": True,
                "queue": [],
                "extra_orders": 0,
                "prompts_max": 1,
                "repeat_no": 1,
                "repeat_max": 1,
                "window_no": 1,
                "total_windows": 1,
                "model_filename": self.t2v_transformer_filename,
            },
            "loras": [],
            "refresh": 0,
            "validate_success": 1,
            "apply_success": 1,
            "advanced": "",
            "model_filename": self.t2v_transformer_filename,
        }
        
        # Generate video using the generate_video function
        generate_video(
            task_id=0,  # Not used in this context
            send_cmd=send_cmd,  # Use our send_cmd function
            prompt=input_data.prompt,
            negative_prompt=input_data.negative_prompt,
            resolution=f"{width}x{height}",
            video_length=input_data.num_frames,
            seed=input_data.seed,
            num_inference_steps=input_data.num_inference_steps,
            guidance_scale=input_data.guidance_scale,
            flow_shift=input_data.shift,
            embedded_guidance_scale=0.0,  # Default value
            repeat_generation=1,  # Default value
            multi_images_gen_type=0,  # Default value
            tea_cache_setting=input_data.tea_cache,
            tea_cache_start_step_perc=input_data.tea_cache_start_step_perc,
            activated_loras=[],  # No loras by default
            loras_multipliers="",  # No loras by default
            image_prompt_type=0,  # Text-to-video mode
            image_start=None,  # Not used for text-to-video
            image_end=None,  # Not used for text-to-video
            model_mode="t2v",  # Text-to-video mode
            video_source=None,  # Not used for text-to-video
            keep_frames_video_source=None,  # Not used for text-to-video
            video_prompt_type="",  # Not used for text-to-video
            image_refs=None,  # Not used for text-to-video
            video_guide=None,  # Not used for text-to-video
            keep_frames_video_guide=None,  # Not used for text-to-video
            video_mask=None,  # Not used for text-to-video
            sliding_window_size=input_data.num_frames,  # Same as video_length
            sliding_window_overlap=0,  # Default value
            sliding_window_discard_last_frames=0,  # Default value
            remove_background_image_ref=False,  # Default value
            temporal_upsampling=input_data.temporal_upsampling,
            spatial_upsampling=input_data.spatial_upsampling,
            RIFLEx_setting=input_data.enable_RIFLEx,
            slg_switch=False,  # Default value
            slg_layers=0,  # Default value
            slg_start_perc=0,  # Default value
            slg_end_perc=0,  # Default value
            cfg_star_switch=input_data.cfg_star_switch,
            cfg_zero_step=input_data.cfg_zero_step,
            state=state,  # Pass the state dictionary
            model_filename=os.path.join(self.ckpts_dir, self.t2v_transformer_filename)  # Include full path to model file
        )
        
        # Find the most recently created MP4 file in the outputs directory
        outputs_dir = os.path.join(str(current_dir), "outputs")
        os.makedirs(outputs_dir, exist_ok=True)
        
        # Get all MP4 files in the outputs directory
        mp4_files = [f for f in os.listdir(outputs_dir) if f.endswith('.mp4')]
        if not mp4_files:
            raise FileNotFoundError("No MP4 files found in outputs directory")
            
        # Get the most recently modified file
        latest_file = max(mp4_files, key=lambda f: os.path.getmtime(os.path.join(outputs_dir, f)))
        latest_file_path = os.path.join(outputs_dir, latest_file)
        
        return AppOutput(video=File(path=latest_file_path))

    async def unload(self):
        """Clean up resources."""
        # Free up GPU memory
        torch.cuda.empty_cache() 

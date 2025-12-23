import os
import sys
from pathlib import Path

# Performance mode configuration
# Set to True for high performance (lower quality) mode
# Set to False for high quality (lower performance) mode
PERFORMANCE_MODE = False

# Add the current directory to Python path
current_dir = Path(__file__).parent.absolute()
sys.path.append(os.path.join(str(current_dir), "wan"))
sys.path.append(os.path.join(str(current_dir), "wan", "wan"))

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
import torch
import tempfile
from typing import Optional, List, Tuple, Union
from pydantic import Field
import numpy as np
import subprocess
from huggingface_hub import hf_hub_download, snapshot_download
import shutil
import requests
from PIL import Image
import os

from .wan.wan.configs import WAN_CONFIGS
from .wan.wan.image2video import WanI2V
from mmgp import offload


class AppInput(BaseAppInput):
    prompt: str = Field(description="Text prompt for video generation")
    input_image: File = Field(description="Input image for image-to-video generation")
    end_frame: Optional[File] = Field(default=None, description="Optional end frame image for video generation")
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
    # Additional parameters for image-to-video
    cfg_star_switch: bool = Field(default=True, description="Enable CFG* guidance")
    cfg_zero_step: int = Field(default=5, description="Step at which to switch to CFG* guidance")
    add_frames_for_end_image: bool = Field(default=True, description="Add frames for end image in image-to-video")


class AppOutput(BaseAppOutput):
    video: File = Field(description="Generated video file")



class App(BaseApp):
    async def setup(self):
        """Initialize the app and download required models"""
        # Initialize model filenames with default values
        if PERFORMANCE_MODE:
            self.i2v_transformer_filename = "wan2.1_image2video_480p_14B_quanto_int8.safetensors"
            self.text_encoder_filename = "models_t5_umt5-xxl-enc-quanto_int8.safetensors"
            self.profile_type = 4  # Use profile 4 for high performance
        else:
            self.i2v_transformer_filename = "wan2.1_image2video_480p_14B_bf16.safetensors"
            self.text_encoder_filename = "models_t5_umt5-xxl-enc-bf16.safetensors"
            self.profile_type = 2  # Use profile 2 for high quality
        
        self.vae_filename = "Wan2.1_VAE.safetensors"
        
        # Create directories for models and loras
        os.makedirs("data", exist_ok=True)
        os.makedirs("loras", exist_ok=True)

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
            
        # Adjust model filename based on dtype
        if self.default_dtype == torch.float16 and "quanto" in self.i2v_transformer_filename:
            self.i2v_transformer_filename = self.i2v_transformer_filename.replace("quanto_int8", "quanto_fp16_int8")
        
        # Store configuration
        self.WAN_CONFIGS = WAN_CONFIGS
        self.offload = offload
        
        # Download models
        self.download_models()
        
        # Initialize the model
        print("Loading model...")
        print("Using image-to-video model...")
        config = self.WAN_CONFIGS['i2v-14B']
        self.wan_model = WanI2V(
            config=config,
            checkpoint_dir=str(self.checkpoint_dir),
            model_filename=self.i2v_transformer_path,
            text_encoder_filename=self.text_encoder_path,
            quantizeTransformer=False,  # Default to non-quantized
            dtype=self.default_dtype,
            VAE_dtype=torch.float32,
            mixed_precision_transformer=False
        )
        
        # This is needed because Wan2GP gradio app has a _interrupt attribute which the original Wan does not have
        self.wan_model._interrupt = False
        
        # Create pipe for offload and Lora support
        self.pipe = {
            "transformer": self.wan_model.model,
            "text_encoder": self.wan_model.text_encoder.model,
            "text_encoder_2": self.wan_model.clip.model,  # Added CLIP model
            "vae": self.wan_model.vae.model
        }
        
        # Configure offloading based on profile
        kwargs = {"extraModelsToQuantize": None}
        if self.profile_type == 2 or self.profile_type == 4:
            kwargs["budgets"] = {
                "transformer": 100,
                "text_encoder": 100,  # Specific budget for text encoder
                "text_encoder_2": 100,  # Specific budget for CLIP model
                "*": 1000
            }
            if self.profile_type == 4:
                kwargs["partialPinning"] = True
        elif self.profile_type == 3:
            kwargs["budgets"] = {"*": "70%"}
            
        # Setup memory profile with more comprehensive settings
        self.offloadobj = self.offload.profile(
            self.pipe, 
            profile_no=self.profile_type,
            compile="",
            quantizeTransformer=False,
            loras="transformer",
            coTenantsMap={},
            perc_reserved_mem_max=0.9,  # Default to 90% reserved memory
            convertWeightsFloatTo=self.default_dtype,
            **kwargs
        )
        
        # Set default device if specified
        if self.device_id >= 0:
            torch.set_default_device(self.device)

    def download_models(self):
        """Download required model files from HuggingFace"""
        from huggingface_hub import hf_hub_download, snapshot_download
        from pathlib import Path
        
        # Create checkpoint directory if it doesn't exist
        self.checkpoint_dir = Path("data")
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Download models from HuggingFace
        repo_id = "DeepBeepMeep/Wan2.1"
        
        # Download main models
        file_list = [
            self.i2v_transformer_filename,
            self.text_encoder_filename,
            self.vae_filename,
            "models_clip_open-clip-xlm-roberta-large-vit-huge-14-bf16.safetensors",
            "flownet.pkl"
        ]
        
        # Download each file if it doesn't exist
        for filename in file_list:
            target_path = self.checkpoint_dir / filename
            if not target_path.exists():
                print(f"Downloading {filename}...")
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=str(self.checkpoint_dir)
                )
        
        # Download additional resources
        source_folders = ["xlm-roberta-large", "pose", "depth", "mask"]
        for folder in source_folders:
            target_folder = self.checkpoint_dir / folder
            if not target_folder.exists():
                print(f"Downloading {folder} resources...")
                snapshot_download(
                    repo_id=repo_id,
                    allow_patterns=f"{folder}/*",
                    local_dir=str(self.checkpoint_dir)
                )
        
        # Set full paths for model files using Path
        self.i2v_transformer_path = str(self.checkpoint_dir / self.i2v_transformer_filename)
        self.text_encoder_path = str(self.checkpoint_dir / self.text_encoder_filename)
        self.vae_path = str(self.checkpoint_dir / self.vae_filename)
        
        # Clean up old format files if they exist
        to_remove = ["models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", "Wan2.1_VAE.pth"]
        for file_name in to_remove:
            file_path = self.checkpoint_dir / file_name
            if file_path.exists():
                try:
                    file_path.unlink()
                except:
                    pass

    async def run(self, input_data: AppInput) -> AppOutput:
        """Run video generation with Wan2GP optimizations."""
        # Parse size
        width, height = map(int, input_data.size.split('x'))
        size = (width, height)
        
        offload.shared_state["_attention"] = input_data.attention
        self.wan_model.model.teacache_skipped_steps = 0
        
        # Handle Lora if provided
        loras = []
        if input_data.lora_file:
            lora_path = f"loras/user_lora.safetensors"
            os.makedirs(os.path.dirname(lora_path), exist_ok=True)
            
            # Download the lora file from URL using requests
            print(f"Downloading Lora from {input_data.lora_file}...")
            try:
                response = requests.get(input_data.lora_file, stream=True)
                response.raise_for_status()  # Raise an exception for HTTP errors
                
                with open(lora_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        
                loras.append(lora_path)
                
                # Load Lora into model
                offload.load_loras_into_model(self.wan_model.model, loras, activate_all_loras=True, lora_multi=torch.tensor([float(input_data.lora_multiplier)], 
                                                                device=self.device), verboseLevel=1)
                
            except Exception as e:
                print(f"Error downloading or loading Lora: {e}")
        
        # Setup TeaCache parameters
        if input_data.tea_cache > 0:
            self.wan_model.model.enable_teacache = True
            self.wan_model.model.teacache_multiplier = input_data.tea_cache
            self.wan_model.model.teacache_start_step = int(input_data.tea_cache_start_step_perc * input_data.num_inference_steps / 100)
            self.wan_model.model.num_steps = input_data.num_inference_steps
            
            # Configure TeaCache coefficients for image-to-video
            self.wan_model.model.coefficients = [-3.02331670e+02, 2.23948934e+02, -5.25463970e+01, 5.87348440e+00, -2.01973289e-01]
        else:
            self.wan_model.model.enable_teacache = False
        
        print(f"Generating video for prompt: {input_data.prompt}")
        print(f"Size: {size}, Frames: {input_data.num_frames}, Steps: {input_data.num_inference_steps}")
        
        # Define progress callback for the UI
        def callback(step_idx, latent, force_refresh, read_state = False, override_num_inference_steps = -1):
            return
        
        # Load and preprocess input image
        input_img = Image.open(input_data.input_image.path).convert("RGB")
        end_img = Image.open(input_data.end_frame.path).convert("RGB") if input_data.end_frame else None
        
        video_tensor = self.wan_model.generate(
            input_prompt=input_data.prompt,
            img=input_img,
            img2=end_img,
            frame_num=input_data.num_frames,
            shift=input_data.shift,
            sample_solver=input_data.sample_solver,
            sampling_steps=input_data.num_inference_steps,
            guide_scale=input_data.guidance_scale,
            n_prompt=input_data.negative_prompt,
            seed=input_data.seed,
            callback=callback,
            enable_RIFLEx=input_data.enable_RIFLEx,
            VAE_tile_size=input_data.vae_tile_size,
            joint_pass=input_data.joint_pass,
            cfg_star_switch=input_data.cfg_star_switch,
            cfg_zero_step=input_data.cfg_zero_step,
            add_frames_for_end_image=input_data.add_frames_for_end_image
        )
        
        # Clean up
        self.offloadobj.release()
        torch.cuda.empty_cache()
        
        if video_tensor is None:
            raise ValueError("Video generation was interrupted or failed")
        
        # Convert tensor to numpy frames
        video_np = video_tensor.permute(1, 2, 3, 0).cpu().numpy()
        video_np = ((video_np + 1) / 2 * 255).astype(np.uint8)
        
        # Create a temporary file for the video
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            output_path = temp_file.name
        
        # Save video using FFmpeg with Safari-specific compatibility settings
        print(f"Saving video to {output_path}...")
        # First save frames as temporary PNG files
        temp_dir = tempfile.mkdtemp()
        for i, frame in enumerate(video_np):
            frame_path = os.path.join(temp_dir, f"frame_{i:05d}.png")
            Image.fromarray(frame).save(frame_path)

        # Use FFmpeg with proven Safari compatibility settings
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-framerate", str(input_data.fps),
            "-i", os.path.join(temp_dir, "frame_%05d.png"),
            "-c:v", "libx264",
            "-profile:v", "main",  # Critical for Safari
            "-pix_fmt", "yuv420p",  # Critical for Safari
            "-movflags", "+faststart",  # Helps with streaming
            "-crf", "23",  # Reasonable quality
            output_path
        ]
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)

        # Clean up temporary files
        shutil.rmtree(temp_dir)
        
        return AppOutput(video=File(path=output_path))


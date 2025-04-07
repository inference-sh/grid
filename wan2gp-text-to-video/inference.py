from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
import os
import torch
import sys
import tempfile
from typing import Optional, List, Tuple, Union
from pydantic import Field
import numpy as np
import subprocess
from moviepy.editor import ImageSequenceClip
from huggingface_hub import snapshot_download, hf_hub_download
import shutil
import requests
from PIL import Image

from mmgp import offload, safetensors2, profile_type
from .wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS, SUPPORTED_SIZES
from .wan.modules.attention import get_attention_modes
from .wan import WanT2V
from mmgp import offload

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
    quantize_transformer: bool = Field(default=True, description="Quantize transformer to 8-bit for lower VRAM usage")
    attention: str = Field(
        default="sage2", 
        description="Attention mechanism to use for generation",
        enum=["auto", "sdpa", "sage2", "flash"]
    )
    

class AppOutput(BaseAppOutput):
    video: File = Field(description="Generated video file")


class App(BaseApp):
    async def setup(self):
       
        # Create directories for models and loras
        os.makedirs("ckpts", exist_ok=True)
        os.makedirs("loras", exist_ok=True)

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_id = 0 if torch.cuda.is_available() else -1
        
        # Store configuration
        self.WAN_CONFIGS = WAN_CONFIGS
        self.offload = offload
        self.profile_type = profile_type
        
        # Download models
        self.download_models()
        
        print("Setup completed successfully!")
        
    def download_models(self):
        """Download required model files from HuggingFace"""
        # Define model files to download
        repo_id = "DeepBeepMeep/Wan2.1"

        self.transformer_filename = hf_hub_download(repo_id=repo_id, filename="wan2.1_text2video_14B_quanto_int8.safetensors")
        self.text_encoder_filename = hf_hub_download(repo_id=repo_id, filename="models_t5_umt5-xxl-enc-quanto_int8.safetensors")

        self.vae_filename = hf_hub_download(repo_id=repo_id, filename="Wan2.1_VAE_bf16.safetensors")
        self.clip_filename = hf_hub_download(repo_id=repo_id, filename="models_clip_open-clip-xlm-roberta-large-vit-huge-14-bf16.safetensors")

        print("All model files downloaded successfully!")

    async def run(self, input_data: AppInput) -> AppOutput:
        """Run video generation with Wan2GP optimizations."""
        # Import necessary modules
        
        # Parse size
        width, height = map(int, input_data.size.split('x'))
        size = (width, height)
        
        # Load model with optimizations
        print("Loading model...")
        checkpoint_dir = "/".join(self.vae_filename.split("/")[:-1])
        print(self.transformer_filename)
        print(self.text_encoder_filename)
        print(self.vae_filename)
        print(self.clip_filename)
        print(checkpoint_dir)
        config = self.WAN_CONFIGS['t2v-14B']
        
        # Initialize the model
        wan_model = WanT2V(
            config=config,
            checkpoint_dir=checkpoint_dir,
            device_id=self.device_id,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_usp=False,
            model_filename=self.transformer_filename,
            text_encoder_filename=self.text_encoder_filename
        )

        # This is needed because Wan2GP gradio app has a _interrupt attribute which the original Wan does not have
        wan_model._interrupt = False

        # TODO get a better solution for this. Possible choices are ["auto", "sdpa", "sage", "sage2", "flash", "xformers"]
        # Sage2 would be the fastest but it's not installed by default.

        offload.shared_state["_attention"] = input_data.attention
        wan_model.teacache_skipped_steps = 0
        wan_model.model.teacache_skipped_steps = 0
        
        # Create pipe for offload and Lora support
        pipe = {
            "transformer": wan_model.model,
            "text_encoder": wan_model.text_encoder.model,
            "vae": wan_model.vae.model
        }
        
        # Configure offloading based on profile
        kwargs = {"extraModelsToQuantize": None}
        kwargs["budgets"] = {
            "transformer": 100,
            "text_encoder": 100,
            "*": 1000
        }
        
        # Setup memory profile
        offloadobj = self.offload.profile(
            pipe, 
            profile_no=4,  # LowRAM_LowVRAM profile
            compile="",
            quantizeTransformer=input_data.quantize_transformer,
            **kwargs
        )
        
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
                offload.load_loras_into_model(wan_model.model, loras, activate_all_loras=True, lora_multi=torch.tensor([float(input_data.lora_multiplier)], 
                                                                    device=self.device), verboseLevel=1)
                
                
            except Exception as e:
                print(f"Error downloading or loading Lora: {e}")
        
        # Setup TeaCache parameters
        if input_data.tea_cache > 0:
            wan_model.model.enable_teacache = True
            wan_model.model.teacache_multiplier = input_data.tea_cache
            wan_model.model.teacache_start_step = int(input_data.tea_cache_start_step_perc * input_data.num_inference_steps / 100)
            wan_model.model.num_steps = input_data.num_inference_steps
            
            # Configure TeaCache coefficients based on model
            wan_model.model.coefficients = [-5784.54975374, 5449.50911966, -1811.16591783, 256.27178429, -13.02252404]
        else:
            wan_model.model.enable_teacache = False
        
        print(f"Generating video for prompt: {input_data.prompt}")
        print(f"Size: {size}, Frames: {input_data.num_frames}, Steps: {input_data.num_inference_steps}")
        
        # Define progress callback for the UI
        def callback(step_idx, latents):
            if step_idx == -1:
                print("Preparing for generation...")
            else:
                print(f"Step {step_idx + 1}/{input_data.num_inference_steps}")
        
        # Generate the video
        video_tensor = wan_model.generate(
            input_prompt=input_data.prompt,
            size=size,
            frame_num=input_data.num_frames,
            shift=input_data.shift,
            sample_solver=input_data.sample_solver,
            sampling_steps=input_data.num_inference_steps,
            guide_scale=input_data.guidance_scale,
            n_prompt=input_data.negative_prompt,
            seed=input_data.seed,
            offload_model=False,
            callback=callback,
            enable_RIFLEx=input_data.enable_RIFLEx,
            VAE_tile_size=input_data.vae_tile_size,
            joint_pass=input_data.joint_pass
        )
        
        # Clean up
        offloadobj.release()
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

    async def unload(self):
        """Clean up resources."""
        # Free up GPU memory
        torch.cuda.empty_cache()
        
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
import os
import torch
import sys
import tempfile
from typing import Optional, List, Tuple, Literal
from pydantic import Field
import numpy as np
import subprocess
from moviepy.editor import ImageSequenceClip
from huggingface_hub import snapshot_download
from .wan.text2video import WanT2V
from .wan.configs import WAN_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES
import random


class AppInput(BaseAppInput):
    prompt: str = Field(description="Text prompt for video generation")
    size: Literal["1280*720", "720*1280", "480*832", "832*480", "1024*1024"] = Field(
        default="1280*720", 
        description="Size of the generated video (width*height)"
    )
    num_frames: int = Field(
        default=81, 
        description="Number of frames to generate (should be 4n+1)"
    )
    fps: int = Field(
        default=8, 
        description="Frames per second for the output video"
    )
    guidance_scale: float = Field(default=9.0, description="Classifier-free guidance scale")
    num_inference_steps: int = Field(default=20, description="Number of denoising steps")
    seed: Optional[int] = Field(default=-1, description="Random seed for reproducibility (-1 for random)")
    negative_prompt: str = Field(default="", description="Negative prompt to guide generation")
    sample_solver: str = Field(default="unipc", description="Solver to use for sampling (unipc or dpm++)")
    shift: float = Field(default=5.0, description="Noise schedule shift parameter")

    class Config:
        schema_extra = {
            "required": ["prompt", "size", "num_frames", "fps"]
        }

class AppOutput(BaseAppOutput):
    video: File = Field(description="Generated video file")

class App(BaseApp):
    async def setup(self):
        # Download Wan2.1 model
        model_path = snapshot_download("Wan-AI/Wan2.1-T2V-14B")
        print(f"Model downloaded to {model_path}")

        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_id = 0 if torch.cuda.is_available() else -1
        
        # Load model configuration
        self.config = WAN_CONFIGS['t2v-14B']
        self.size_configs = SIZE_CONFIGS
        
        # Initialize the model
        self.model = WanT2V(
            config=self.config,
            checkpoint_dir=model_path,
            device_id=self.device_id,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_usp=False,
            t5_cpu=True
        )
        print("Model loaded successfully!")

    async def run(self, input_data: AppInput) -> AppOutput:
        """Run video generation on the input prompt."""
        # Validate inputs
        task = "t2v-14B"  # We're using the text-to-video model
        
        # Validate size
        if input_data.size not in SUPPORTED_SIZES[task]:
            supported_sizes = ", ".join(SUPPORTED_SIZES[task])
            raise ValueError(f"Unsupported size: {input_data.size}. Supported sizes for {task}: {supported_sizes}")
        
        # Validate frame number (should be 4n+1)
        if (input_data.num_frames - 1) % 4 != 0:
            raise ValueError(f"Number of frames should be 4n+1, got {input_data.num_frames}")
        
        # Set seed
        seed = input_data.seed if input_data.seed >= 0 else random.randint(0, sys.maxsize)
        
        # Generate the video
        print(f"Generating video for prompt: {input_data.prompt}")
        print(f"Size: {input_data.size}, Frames: {input_data.num_frames}, Steps: {input_data.num_inference_steps}")
        
        video_tensor = self.model.generate(
            input_prompt=input_data.prompt,
            size=SIZE_CONFIGS[input_data.size],
            frame_num=input_data.num_frames,
            shift=input_data.shift,
            sample_solver=input_data.sample_solver,
            sampling_steps=input_data.num_inference_steps,
            guide_scale=input_data.guidance_scale,
            n_prompt=input_data.negative_prompt,
            seed=seed,
            offload_model=False
        )
        
        # Convert tensor to numpy frames
        # The tensor is in format [C, F, H, W] with values in range [-1, 1]
        video_np = (video_tensor.permute(1, 2, 3, 0).cpu().numpy() + 1) / 2 * 255
        video_np = video_np.astype(np.uint8)
        
        # Create a temporary file for the video
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            output_path = temp_file.name
        
        # Create and save the video
        clip = ImageSequenceClip([frame for frame in video_np], fps=input_data.fps)
        clip.write_videofile(output_path, codec="libx264", verbose=False, logger=None)
        
        return AppOutput(video=File(path=output_path))

    async def unload(self):
        """Clean up resources here."""
        # Free up GPU memory
        if hasattr(self, 'model'):
            del self.model
            torch.cuda.empty_cache()
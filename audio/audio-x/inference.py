import os
import sys
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from typing import List, Optional
import torch
import torchaudio
from einops import rearrange
import gc
from .stable_audio_tools import get_pretrained_model
from .stable_audio_tools.inference.generation import generate_diffusion_cond
from .stable_audio_tools.data.utils import read_video, merge_video_audio, load_and_process_audio
import stat
import platform
import logging
import tempfile
from transformers import logging as transformers_logging

transformers_logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)

class AppInput(BaseAppInput):
    prompt: str = Field(description="The text prompt for audio generation")
    negative_prompt: Optional[str] = Field(None, description="Negative prompt for generation")
    video_file: Optional[File] = Field(None, description="Input video file for conditioning")
    audio_prompt_file: Optional[File] = Field(None, description="Input audio file for conditioning")
    seconds_start: int = Field(0, description="Start time in seconds")
    seconds_total: int = Field(10, description="Total duration in seconds")
    cfg_scale: float = Field(7.0, description="Classifier-free guidance scale")
    steps: int = Field(100, description="Number of diffusion steps")
    seed: int = Field(-1, description="Random seed (-1 for random)")
    sampler_type: str = Field("dpmpp-3m-sde", description="Sampler type")
    sigma_min: float = Field(0.03, description="Minimum sigma value")
    sigma_max: float = Field(500, description="Maximum sigma value")
    cfg_rescale: float = Field(0.0, description="CFG rescale amount")

class AppOutput(BaseAppOutput):
    audio_output: File = Field(description="Generated audio file")
    video_output: Optional[File] = Field(None, description="Generated video file with audio")

class App(BaseApp):
    async def setup(self, metadata):
        """Initialize the AudioX model and resources."""
        print("Initializing AudioX model...")
        self.model, self.model_config = get_pretrained_model('HKUSTAudio/AudioX')
        self.sample_rate = self.model_config["sample_rate"]
        self.sample_size = self.model_config["sample_size"]
        
        # Create temp directory for processing
        self.temp_dir = tempfile.mkdtemp(prefix="audiox_")
        os.chmod(self.temp_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        
        print("AudioX model initialized successfully")

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Generate audio using AudioX."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # Setup device
        try:
            has_mps = platform.system() == "Darwin" and torch.backends.mps.is_available()
        except Exception:
            has_mps = False
        if has_mps:
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        
        self.model = self.model.to(device)

        # Process video input if provided
        video_path = input_data.video_file.path if input_data.video_file is not None else None
        target_fps = self.model_config.get("video_fps", 5)
        video_tensors = read_video(video_path, seek_time=input_data.seconds_start, 
                                 duration=input_data.seconds_total, target_fps=target_fps)

        # Process audio prompt if provided
        audio_path = input_data.audio_prompt_file.path if input_data.audio_prompt_file is not None else None
        audio_tensor = load_and_process_audio(audio_path, self.sample_rate, 
                                            input_data.seconds_start, input_data.seconds_total)
        audio_tensor = audio_tensor.to(device) if audio_tensor is not None else None

        # Prepare conditioning
        seconds_input = self.sample_size / self.sample_rate
        conditioning = [{
            "video_prompt": [video_tensors.unsqueeze(0)] if video_tensors is not None else None,
            "text_prompt": input_data.prompt,
            "audio_prompt": audio_tensor.unsqueeze(0) if audio_tensor is not None else None,
            "seconds_start": input_data.seconds_start,
            "seconds_total": seconds_input
        }]

        # Prepare negative conditioning if provided
        if input_data.negative_prompt:
            negative_conditioning = [{
                "video_prompt": [video_tensors.unsqueeze(0)] if video_tensors is not None else None,
                "text_prompt": input_data.negative_prompt,
                "audio_prompt": audio_tensor.unsqueeze(0) if audio_tensor is not None else None,
                "seconds_start": input_data.seconds_start,
                "seconds_total": input_data.seconds_total
            }] * 1
        else:
            negative_conditioning = None

        # Generate audio
        audio = generate_diffusion_cond(
            self.model,
            conditioning=conditioning,
            negative_conditioning=negative_conditioning,
            steps=input_data.steps,
            cfg_scale=input_data.cfg_scale,
            batch_size=1,
            sample_size=self.sample_size,
            sample_rate=self.sample_rate,
            seed=input_data.seed,
            device=device,
            sampler_type=input_data.sampler_type,
            sigma_min=input_data.sigma_min,
            sigma_max=input_data.sigma_max,
            init_audio=None,
            init_noise_level=0.1,
            mask_args=None,
            callback=None,
            scale_phi=input_data.cfg_rescale
        )

        # Process generated audio
        audio = rearrange(audio, "b d n -> d (b n)")
        samples_10s = 10 * self.sample_rate
        audio = audio[:, :samples_10s]
        audio = audio.to(torch.float32).div(torch.max(torch.abs(audio))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

        # Save audio output
        output_audio_path = os.path.join(self.temp_dir, "output.wav")
        torchaudio.save(output_audio_path, audio, self.sample_rate)

        # Process video output if video input was provided
        output_video_path = None
        if input_data.video_file is not None:
            output_video_path = os.path.join(self.temp_dir, f"output_{os.path.basename(video_path)}")
            merge_video_audio(
                video_path,
                output_audio_path,
                output_video_path,
                input_data.seconds_start,
                input_data.seconds_total
            )

        # Cleanup
        del video_tensors
        torch.cuda.empty_cache()
        gc.collect()

        return AppOutput(
            audio_output=File(path=output_audio_path),
            video_output=File(path=output_video_path) if output_video_path else None
        )

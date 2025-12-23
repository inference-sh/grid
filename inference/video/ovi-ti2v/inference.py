import os
import sys
import torch
import tempfile
import logging
from accelerate import Accelerator
from huggingface_hub import snapshot_download
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from typing import Optional

# Enable faster HuggingFace downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Add current directory to Python path for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'Ovi'))

# Change working directory to Ovi so relative paths work
ovi_dir = os.path.join(current_dir, 'Ovi')
original_cwd = os.getcwd()
os.chdir(ovi_dir)

from ovi.ovi_fusion_engine import OviFusionEngine
from ovi.utils.io_utils import save_video

# Restore original working directory
os.chdir(original_cwd)


class AppInput(BaseAppInput):
    text_prompt: str = Field(description="Text description for the video. Speech should be wrapped with <S>...<E>. Optional audio description can be wrapped in <AUDCAP>...<ENDAUDCAP>")
    image: Optional[File] = Field(default=None, description="Optional first frame image for image-to-video generation")
    video_frame_height: int = Field(default=512, ge=128, le=1280, description="Video frame height (must be divisible by 32)")
    video_frame_width: int = Field(default=992, ge=128, le=1280, description="Video frame width (must be divisible by 32)")
    seed: int = Field(default=100, ge=0, le=100000, description="Random seed for reproducibility")
    solver_name: str = Field(default="unipc", description="Solver name: unipc, euler, or dpm++")
    sample_steps: int = Field(default=50, ge=20, le=100, description="Number of sampling steps")
    shift: float = Field(default=5.0, ge=0.0, le=20.0, description="Shift parameter")
    video_guidance_scale: float = Field(default=4.0, ge=0.0, le=10.0, description="Video guidance scale (CFG)")
    audio_guidance_scale: float = Field(default=3.0, ge=0.0, le=10.0, description="Audio guidance scale (CFG)")
    slg_layer: int = Field(default=11, ge=-1, le=30, description="SLG layer parameter")
    video_negative_prompt: str = Field(default="", description="Negative prompt for video generation")
    audio_negative_prompt: str = Field(default="", description="Negative prompt for audio generation")

class AppOutput(BaseAppOutput):
    video: File = Field(description="Generated video with audio (.mp4)")
    first_frame: Optional[File] = Field(default=None, description="Generated first frame image if mode is t2i2v (.png)")


class App(BaseApp):
    def download_model_weights(self, ckpt_dir: str):
        """Download required model weights from HuggingFace."""
        logging.info("Checking and downloading model weights...")

        # Download Wan2.2-TI2V-5B (video model and VAE)
        wan_dir = os.path.join(ckpt_dir, "Wan2.2-TI2V-5B")
        logging.info(f"Downloading Wan2.2-TI2V-5B to {wan_dir}")
        snapshot_download(
            repo_id="Wan-AI/Wan2.2-TI2V-5B",
            local_dir=wan_dir,
            local_dir_use_symlinks=False,
            allow_patterns=[
                "google/*",
                "models_t5_umt5-xxl-enc-bf16.pth",
                "Wan2.2_VAE.pth"
            ],
            resume_download=True
        )
        logging.info("✅ Wan2.2-TI2V-5B downloaded")

        # Download MMAudio (audio generation weights)
        mm_audio_dir = os.path.join(ckpt_dir, "MMAudio")
        logging.info(f"Downloading MMAudio to {mm_audio_dir}")
        snapshot_download(
            repo_id="hkchengrex/MMAudio",
            local_dir=mm_audio_dir,
            local_dir_use_symlinks=False,
            allow_patterns=[
                "ext_weights/best_netG.pt",
                "ext_weights/v1-16.pth"
            ],
            resume_download=True
        )
        logging.info("✅ MMAudio downloaded")

        # Download Ovi (fusion model)
        ovi_model_dir = os.path.join(ckpt_dir, "Ovi")
        logging.info(f"Downloading Ovi fusion model to {ovi_model_dir}")
        snapshot_download(
            repo_id="chetwinlow1/Ovi",
            local_dir=ovi_model_dir,
            local_dir_use_symlinks=False,
            allow_patterns=[
                "model.safetensors"
            ],
            resume_download=True
        )
        logging.info("✅ Ovi fusion model downloaded")
        logging.info("All model weights downloaded successfully!")

    async def setup(self, metadata):
        """Initialize OviFusionEngine."""
        # Initialize accelerator for device management
        self.accelerator = Accelerator()
        self.device = self.accelerator.device

        # Determine device_id for compatibility
        if hasattr(self.device, 'index') and self.device.index is not None:
            self.device_id = self.device.index
        else:
            self.device_id = 0 if str(self.device) == 'cuda' else 'cpu'

        logging.info(f"Initializing OviFusionEngine on device: {self.device}")

        # Load default config from yaml
        from omegaconf import OmegaConf
        config_path = os.path.join(current_dir, 'Ovi', 'ovi', 'configs', 'inference', 'inference_fusion.yaml')
        config = OmegaConf.load(config_path)

        # Update ckpt_dir to absolute path
        ckpt_dir = os.path.join(current_dir, 'Ovi', 'ckpts')
        config.ckpt_dir = ckpt_dir

        # Download model weights if not present
        self.download_model_weights(ckpt_dir)

        # Configure model
        config.fp8 = False
        config.cpu_offload = False
        logging.info("Using full precision model (48GB VRAM)")

        # Change to Ovi directory for initialization (relative paths in engine)
        saved_cwd = os.getcwd()
        os.chdir(ovi_dir)

        try:
            # Initialize OviFusionEngine
            self.ovi_engine = OviFusionEngine(
                config=config,
                device=self.device_id,
                target_dtype=torch.bfloat16
            )
            logging.info("OviFusionEngine loaded successfully")
        finally:
            # Restore working directory
            os.chdir(saved_cwd)

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Generate video with audio from text prompt and optional image."""

        # Determine mode and image path
        image_path = None
        mode = "t2v"

        if input_data.image is not None:
            if input_data.image.exists():
                image_path = input_data.image.path
                mode = "i2v"
                logging.info(f"Using image-to-video mode with image: {image_path}")
            else:
                logging.warning("Image file provided but does not exist, falling back to text-to-video mode")

        # Validate solver name
        valid_solvers = ["unipc", "euler", "dpm++"]
        if input_data.solver_name not in valid_solvers:
            raise ValueError(f"Invalid solver_name '{input_data.solver_name}'. Must be one of: {valid_solvers}")

        # Generate video with audio
        logging.info(f"Generating video with mode={mode}, prompt='{input_data.text_prompt[:50]}...'")

        generated_video, generated_audio, generated_image = self.ovi_engine.generate(
            text_prompt=input_data.text_prompt,
            image_path=image_path,
            video_frame_height_width=[input_data.video_frame_height, input_data.video_frame_width],
            seed=input_data.seed,
            solver_name=input_data.solver_name,
            sample_steps=input_data.sample_steps,
            shift=input_data.shift,
            video_guidance_scale=input_data.video_guidance_scale,
            audio_guidance_scale=input_data.audio_guidance_scale,
            slg_layer=input_data.slg_layer,
            video_negative_prompt=input_data.video_negative_prompt,
            audio_negative_prompt=input_data.audio_negative_prompt
        )

        # Save video with audio
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            video_output_path = tmp.name

        save_video(video_output_path, generated_video, generated_audio, fps=24, sample_rate=16000)
        logging.info(f"Video saved to: {video_output_path}")

        # Save generated first frame image if available (t2i2v mode)
        first_frame_path = None
        if generated_image is not None:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                first_frame_path = tmp.name
            generated_image.save(first_frame_path)
            logging.info(f"First frame image saved to: {first_frame_path}")

        return AppOutput(
            video=File(path=video_output_path),
            first_frame=File(path=first_frame_path) if first_frame_path else None
        )


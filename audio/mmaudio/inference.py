import sys
import os
import logging
from pathlib import Path
import torch
import torchaudio
from accelerate import Accelerator
from huggingface_hub import hf_hub_download
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from typing import Optional

# Enable accelerated HF transfers
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Add MMAudio to sys path
current_dir = Path(__file__).parent
mmaudio_path = current_dir / "MMAudio"
if str(mmaudio_path) not in sys.path:
    sys.path.insert(0, str(mmaudio_path))

from mmaudio.eval_utils import generate, load_video, make_video, setup_eval_logging
from mmaudio.model.flow_matching import FlowMatching
from mmaudio.model.networks import MMAudio, get_my_mmaudio
from mmaudio.model.utils.features_utils import FeaturesUtils

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

log = logging.getLogger()

# Hugging Face repository info
HF_REPO_ID = "hkchengrex/MMAudio"

# MMAudio model variants mapping with HF paths
MODEL_VARIANTS = {
    "small_16k": {
        "model_name": "small_16k",
        "model_file": "weights/mmaudio_small_16k.pth",
        "vae_file": "ext_weights/v1-16.pth",
        "bigvgan_file": "ext_weights/best_netG.pt",
        "mode": "16k"
    },
    "small_44k": {
        "model_name": "small_44k", 
        "model_file": "weights/mmaudio_small_44k.pth",
        "vae_file": "ext_weights/v1-44.pth",
        "bigvgan_file": None,
        "mode": "44k"
    },
    "medium_44k": {
        "model_name": "medium_44k",
        "model_file": "weights/mmaudio_medium_44k.pth", 
        "vae_file": "ext_weights/v1-44.pth",
        "bigvgan_file": None,
        "mode": "44k"
    },
    "large_44k": {
        "model_name": "large_44k",
        "model_file": "weights/mmaudio_large_44k.pth",
        "vae_file": "ext_weights/v1-44.pth", 
        "bigvgan_file": None,
        "mode": "44k"
    },
    "default": {
        "model_name": "large_44k_v2",
        "model_file": "weights/mmaudio_large_44k_v2.pth",
        "vae_file": "ext_weights/v1-44.pth",
        "bigvgan_file": None,
        "mode": "44k"
    },
}
DEFAULT_VARIANT = "default"

class ModelConfig:
    """Custom ModelConfig that uses Hugging Face Hub downloads."""
    def __init__(self, variant_info):
        self.model_name = variant_info["model_name"]
        self.mode = variant_info["mode"]
        self.model_file = variant_info["model_file"]
        self.vae_file = variant_info["vae_file"]
        self.bigvgan_file = variant_info["bigvgan_file"]
        self.synchformer_file = "ext_weights/synchformer_state_dict.pth"
        
        # These will be set after download
        self.model_path = None
        self.vae_path = None
        self.bigvgan_16k_path = None
        self.synchformer_ckpt = None
    
    @property
    def seq_cfg(self):
        """Get sequence configuration based on mode."""
        # Import here to avoid circular imports
        from mmaudio.model.sequence_config import CONFIG_16K, CONFIG_44K
        if self.mode == '16k':
            return CONFIG_16K
        elif self.mode == '44k':
            return CONFIG_44K
    
    def download_if_needed(self):
        """Download model files using Hugging Face Hub."""
        log.info(f"Downloading MMAudio model files from {HF_REPO_ID}")
        
        # Download main model
        self.model_path = Path(hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=self.model_file,
            repo_type="model"
        ))
        log.info(f"Downloaded model: {self.model_path}")
        
        # Download VAE
        self.vae_path = Path(hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=self.vae_file,
            repo_type="model"
        ))
        log.info(f"Downloaded VAE: {self.vae_path}")
        
        # Download BigVGAN if needed
        if self.bigvgan_file is not None:
            self.bigvgan_16k_path = Path(hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=self.bigvgan_file,
                repo_type="model"
            ))
            log.info(f"Downloaded BigVGAN: {self.bigvgan_16k_path}")
        
        # Download Synchformer
        self.synchformer_ckpt = Path(hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=self.synchformer_file,
            repo_type="model"
        ))
        log.info(f"Downloaded Synchformer: {self.synchformer_ckpt}")

class AppInput(BaseAppInput):
    prompt: str = Field(description="Text prompt for audio generation")
    negative_prompt: str = Field(default="", description="Negative prompt to avoid certain audio characteristics")
    video_input: Optional[File] = Field(default=None, description="Optional video file to generate synchronized audio for")
    duration: float = Field(default=8.0, description="Duration of audio to generate in seconds")
    cfg_strength: float = Field(default=4.5, description="Classifier-free guidance strength")
    num_steps: int = Field(default=25, description="Number of diffusion steps")
    seed: int = Field(default=42, description="Random seed for reproducible generation")
    mask_away_clip: bool = Field(default=False, description="Whether to mask away CLIP features")

class AppOutput(BaseAppOutput):
    audio_output: File = Field(description="Generated audio file in FLAC format")
    video_output: Optional[File] = Field(default=None, description="Video with generated audio (if video input provided)")
    duration: float = Field(description="Actual duration of generated audio")
    sampling_rate: int = Field(description="Sampling rate of the generated audio")

class App(BaseApp):
    def __init__(self):
        super().__init__()
        self.device = None
        self.net = None
        self.feature_utils = None

    async def setup(self, metadata):
        """Initialize the MMAudio model and resources."""
        setup_eval_logging()
        
        # Initialize accelerator and store device
        accelerator = Accelerator()
        self.device = accelerator.device
        
        log.info(f'Using device: {self.device}')
        
        # Get variant from metadata
        variant = getattr(metadata, "app_variant", DEFAULT_VARIANT)
        if variant not in MODEL_VARIANTS:
            log.warning(f'Unknown model variant: {variant}, falling back to {DEFAULT_VARIANT}')
            variant = DEFAULT_VARIANT
        
        log.info(f'Loading MMAudio model variant: {variant}')
        
        # Create model configuration and download from Hugging Face
        variant_info = MODEL_VARIANTS[variant]
        self.model_config = ModelConfig(variant_info)
        self.model_config.download_if_needed()
        self.seq_cfg = self.model_config.seq_cfg
        
        # Use bfloat16 for efficiency
        self.dtype = torch.bfloat16
        
        # Load the pretrained model
        self.net: MMAudio = get_my_mmaudio(self.model_config.model_name).to(self.device, self.dtype).eval()
        self.net.load_weights(torch.load(self.model_config.model_path, map_location=self.device, weights_only=True))
        
        # Ensure no gradients for inference
        for param in self.net.parameters():
            param.requires_grad = False
        
        log.info(f'Loaded weights from {self.model_config.model_path}')
        
        # Setup feature utils
        self.feature_utils = FeaturesUtils(
            tod_vae_ckpt=self.model_config.vae_path,
            synchformer_ckpt=self.model_config.synchformer_ckpt,
            enable_conditions=True,
            mode=self.model_config.mode,
            bigvgan_vocoder_ckpt=self.model_config.bigvgan_16k_path,
            need_vae_encoder=False
        ).to(self.device, self.dtype).eval()
        
        # Ensure no gradients for feature utils
        for param in self.feature_utils.parameters():
            param.requires_grad = False
        
        log.info('MMAudio model setup complete')

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Generate audio using MMAudio."""
        
        # Setup flow matching for this generation
        rng = torch.Generator(device=self.device)
        rng.manual_seed(input_data.seed)
        fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=input_data.num_steps)
        
        # Process video input if provided
        video_info = None
        clip_frames = None
        sync_frames = None
        duration = input_data.duration
        
        if input_data.video_input is not None:
            if not input_data.video_input.exists():
                raise RuntimeError(f"Video file does not exist at path: {input_data.video_input.path}")
            
            log.info(f'Using video {input_data.video_input.path}')
            video_info = load_video(Path(input_data.video_input.path), duration)
            clip_frames = video_info.clip_frames
            sync_frames = video_info.sync_frames
            duration = video_info.duration_sec
            
            if input_data.mask_away_clip:
                clip_frames = None
            else:
                clip_frames = clip_frames.unsqueeze(0)
            sync_frames = sync_frames.unsqueeze(0)
        else:
            log.info('No video provided -- text-to-audio mode')
        
        # Update sequence configuration
        self.seq_cfg.duration = duration
        self.net.update_seq_lengths(self.seq_cfg.latent_seq_len, self.seq_cfg.clip_seq_len, self.seq_cfg.sync_seq_len)
        
        log.info(f'Prompt: {input_data.prompt}')
        log.info(f'Negative prompt: {input_data.negative_prompt}')
        log.info(f'Duration: {duration} seconds')
        
        # Generate audio with no gradients
        with torch.no_grad():
            audios = generate(
                clip_frames,
                sync_frames, 
                [input_data.prompt],
                negative_text=[input_data.negative_prompt],
                feature_utils=self.feature_utils,
                net=self.net,
                fm=fm,
                rng=rng,
                cfg_strength=input_data.cfg_strength
            )
        
        audio = audios.float().cpu()[0]
        
        # Save audio file
        output_dir = Path("/tmp/mmaudio_output")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if input_data.video_input is not None:
            video_stem = Path(input_data.video_input.filename or "video").stem
            audio_save_path = output_dir / f'{video_stem}_audio.flac'
        else:
            safe_filename = input_data.prompt.replace(' ', '_').replace('/', '_').replace('.', '')[:50]
            audio_save_path = output_dir / f'{safe_filename}.flac'
        
        torchaudio.save(audio_save_path, audio, self.seq_cfg.sampling_rate)
        log.info(f'Audio saved to {audio_save_path}')
        
        # Create video with audio if video input was provided
        video_output_file = None
        if input_data.video_input is not None and video_info is not None:
            video_save_path = output_dir / f'{video_stem}_with_audio.mp4'
            make_video(video_info, video_save_path, audio, sampling_rate=self.seq_cfg.sampling_rate)
            log.info(f'Video with audio saved to {video_save_path}')
            video_output_file = File(path=str(video_save_path))
        
        # Log memory usage
        if self.device.type == 'cuda':
            log.info('Memory usage: %.2f GB', torch.cuda.max_memory_allocated() / (2**30))
        
        return AppOutput(
            audio_output=File(path=str(audio_save_path)),
            video_output=video_output_file,
            duration=duration,
            sampling_rate=self.seq_cfg.sampling_rate
        )

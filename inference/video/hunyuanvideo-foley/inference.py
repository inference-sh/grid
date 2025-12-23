import os
import sys

# Add current directory to Python path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Enable faster HuggingFace downloads globally
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import tempfile
import torch
import torchaudio
from loguru import logger
import random
import numpy as np
from huggingface_hub import snapshot_download
from accelerate import Accelerator

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field

from hunyuanvideo_foley.utils.model_utils import load_model
from hunyuanvideo_foley.utils.feature_utils import feature_process
from hunyuanvideo_foley.utils.model_utils import denoise_process
from hunyuanvideo_foley.utils.media_utils import merge_audio_video


class AppInput(BaseAppInput):
    video: File = Field(description="Input video file for audio generation")
    text_prompt: str = Field(
        default="",
        description="Text description of the desired audio (English). Leave empty for video-only audio generation"
    )
    guidance_scale: float = Field(
        default=4.5,
        ge=1.0,
        le=10.0,
        description="Classifier-free guidance scale (1.0-10.0). Higher values follow the prompt more closely"
    )
    num_inference_steps: int = Field(
        default=50,
        ge=10,
        le=100,
        description="Number of denoising steps (10-100). More steps = higher quality but slower"
    )
    sample_nums: int = Field(
        default=1,
        ge=1,
        le=6,
        description="Number of audio samples to generate (1-6)"
    )


class AppOutput(BaseAppOutput):
    videos_with_audio: list[File] = Field(description="Generated videos with synthesized audio")
    status_message: str = Field(description="Processing status and information")


class App(BaseApp):
    def __init__(self):
        super().__init__()
        self.model_dict = None
        self.cfg = None
        self.accelerator = None
        self.device = None
        
        # Model paths - will be set after downloading
        self.MODEL_PATH = None
        self.CONFIG_PATH = None
        self.repo_id = "tencent/HunyuanVideo-Foley"

    def download_model_from_hf(self) -> str:
        """Download model from HuggingFace using cache"""
        try:
            logger.info(f"Starting download from HuggingFace: {self.repo_id}")
            
            # Download entire repository using HuggingFace cache
            model_path = snapshot_download(
                repo_id=self.repo_id,
                resume_download=True,  # Support resume
                local_files_only=False,  # Allow network downloads
            )
            
            logger.info(f"✅ Model downloaded successfully to cache: {model_path}")
            return model_path
            
        except Exception as e:
            error_msg = f"❌ Model download failed: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)

    async def setup(self, metadata):
        """Initialize models and resources once"""
        try:
            # Set manual seed for reproducibility
            random.seed(1)
            np.random.seed(1)
            torch.manual_seed(1)
            
            # Setup accelerator for proper device management
            self.accelerator = Accelerator()
            self.device = self.accelerator.device
            logger.info(f"Using device: {self.device}")
            
            # Download model from HuggingFace using cache
            logger.info("Downloading model from HuggingFace...")
            model_cache_path = self.download_model_from_hf()
            
            # Set paths based on downloaded cache location
            self.MODEL_PATH = model_cache_path
            self.CONFIG_PATH = os.path.join(model_cache_path, "config.yaml")
            
            # Verify files exist
            if not os.path.exists(self.CONFIG_PATH):
                raise Exception(f"❌ Config file not found: {self.CONFIG_PATH}")
            
            # Load model
            logger.info("Loading model...")
            logger.info(f"Model path: {self.MODEL_PATH}")
            logger.info(f"Config path: {self.CONFIG_PATH}")
            
            self.model_dict, self.cfg = load_model(self.MODEL_PATH, self.CONFIG_PATH, self.device)
            
            logger.info("✅ Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise Exception(f"❌ Model loading failed: {str(e)}")

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Process individual requests"""
        if self.model_dict is None or self.cfg is None:
            raise Exception("❌ Model not loaded. Please check setup method.")
        
        if input_data.video is None:
            raise Exception("❌ Please provide a video file!")
        
        # Allow empty text prompt, use empty string if no prompt provided
        text_prompt = input_data.text_prompt.strip() if input_data.text_prompt else ""
        
        try:
            logger.info(f"Processing video: {input_data.video.path}")
            logger.info(f"Text prompt: {text_prompt}")
            logger.info(f"Generating {input_data.sample_nums} audio samples...")
            
            # Feature processing
            visual_feats, text_feats, audio_len_in_s = feature_process(
                input_data.video.path,
                text_prompt,
                self.model_dict,
                self.cfg
            )
            
            # Denoising process to generate multiple audio samples
            # The model generates sample_nums audio samples per inference
            # The denoise_process function returns audio with shape [batch_size, channels, samples]
            audio, sample_rate = denoise_process(
                visual_feats,
                text_feats,
                audio_len_in_s,
                self.model_dict,
                self.cfg,
                guidance_scale=input_data.guidance_scale,
                num_inference_steps=input_data.num_inference_steps,
                batch_size=input_data.sample_nums
            )
            
            # Create temporary files to save results
            temp_dir = tempfile.mkdtemp()
            video_outputs = []
            
            # Process each generated audio sample
            for i in range(input_data.sample_nums):
                # Save audio file
                audio_output = os.path.join(temp_dir, f"generated_audio_{i+1}.wav")
                torchaudio.save(audio_output, audio[i], sample_rate)
                
                # Merge video and audio
                video_output = os.path.join(temp_dir, f"video_with_audio_{i+1}.mp4")
                merge_audio_video(audio_output, input_data.video.path, video_output)
                
                # Create File object for output
                video_file = File(path=video_output)
                video_outputs.append(video_file)
            
            status_message = f"✅ Generated {input_data.sample_nums} audio sample(s) successfully!"
            logger.info("Inference completed!")
            
            return AppOutput(
                videos_with_audio=video_outputs,
                status_message=status_message
            )
            
        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            raise Exception(f"❌ Inference failed: {str(e)}")
import os
import sys
import tempfile

# Enable HF Transfer for faster downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Force flash_attn to be used instead of SDPA
os.environ["TRANSFORMERS_ATTENTION_TYPE"] = "flash_attention_2"

# Smart path setup for InfiniteTalk imports to work in any environment
base_path = os.path.dirname(__file__)
infinitetalk_path = os.path.join(base_path, 'InfiniteTalk')

# Add all necessary paths for InfiniteTalk imports to work
paths_to_add = [
    base_path,                                      # For 'from InfiniteTalk import'
    infinitetalk_path,                              # For internal InfiniteTalk imports  
    os.path.join(infinitetalk_path, 'src'),         # For 'from src.vram_management import'
    os.path.join(infinitetalk_path, 'wan'),         # For 'from wan.utils import'
]

for path in paths_to_add:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)

import warnings
import logging
import torch
import torch.distributed as dist
import numpy as np
import librosa
import pyloudnorm as pyln
import soundfile as sf
import re
from datetime import datetime
from PIL import Image
from einops import rearrange
from transformers import Wav2Vec2FeatureExtractor
from huggingface_hub import hf_hub_download, snapshot_download

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from typing import List, Optional

# Import from InfiniteTalk using relative imports
from .InfiniteTalk import wan
from .InfiniteTalk.wan.configs import SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from .InfiniteTalk.wan.utils.utils import cache_image, cache_video, str2bool
from .InfiniteTalk.wan.utils.multitalk_utils import save_video_ffmpeg
from .InfiniteTalk.src.audio_analysis.wav2vec2 import Wav2Vec2Model

warnings.filterwarnings('ignore')

class AppInput(BaseAppInput):
    image: File = Field(description="Input image for image-to-video generation")
    prompt: str = Field(description="Text prompt describing the desired video")
    negative_prompt: str = Field(
        default="bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
        description="Negative prompt"
    )
    audio_file: File = Field(description="Audio file to drive the video generation")
    resolution: str = Field(
        default="infinitetalk-480",
        description="Output resolution",
        enum=["infinitetalk-480", "infinitetalk-720"]
    )
    sampling_steps: int = Field(default=8, description="Number of sampling steps")
    seed: int = Field(default=42, description="Random seed")
    text_guide_scale: float = Field(default=1.0, description="Text guidance scale")
    audio_guide_scale: float = Field(default=2.0, description="Audio guidance scale")
    use_teacache: bool = Field(default=False, description="Enable TeaCache for faster inference")
    teacache_thresh: float = Field(default=0.1, description="TeaCache threshold value")

class AppOutput(BaseAppOutput):
    video: File = Field(description="Generated video file")



class App(BaseApp):
    async def setup(self, metadata):
        """Initialize the InfiniteTalk model and resources."""
        self.rank = int(os.getenv("RANK", 0))
        self.world_size = int(os.getenv("WORLD_SIZE", 1))
        self.local_rank = int(os.getenv("LOCAL_RANK", 0))
        
        # Initialize accelerator first
        from accelerate import Accelerator
        self.accelerator = Accelerator()
        
        # Always use accelerator device directly - no manual device strings
        self.device = self.accelerator.device
        
        # For components that need device_id as integer, extract it from accelerator
        if hasattr(self.device, 'index') and self.device.index is not None:
            self.device_id = self.device.index
        elif str(self.device).startswith('cuda:'):
            self.device_id = int(str(self.device).split(':')[1])
        elif str(self.device) == 'cuda':
            self.device_id = 0
        else:
            self.device_id = 'cpu'
        
        # Setup logging
        if self.rank == 0:
            logging.basicConfig(
                level=logging.INFO,
                format="[%(asctime)s] %(levelname)s: %(message)s",
                handlers=[logging.StreamHandler(stream=sys.stdout)]
            )
        else:
            logging.basicConfig(level=logging.ERROR)
            
        # Initialize distributed if needed
        if self.world_size > 1:
            if self.device.type == 'cuda':
                torch.cuda.set_device(self.local_rank)
            dist.init_process_group(
                backend="nccl" if self.device.type == 'cuda' else "gloo",
                init_method="env://",
                rank=self.rank,
                world_size=self.world_size
            )
        
        # Download required models using HuggingFace hub (cached by default)
        logging.info("Downloading required model weights...")
        
        # Download Wan2.1-I2V-14B-480P model
        self.ckpt_dir = snapshot_download("Wan-AI/Wan2.1-I2V-14B-480P")
        
        # Download chinese-wav2vec2-base model
        self.wav2vec_dir = snapshot_download("TencentGameMate/chinese-wav2vec2-base")
        
        # Download additional model.safetensors from PR
        hf_hub_download("TencentGameMate/chinese-wav2vec2-base", 
                       "model.safetensors", revision="refs/pr/1")
        
        # Download InfiniteTalk model
        infinitetalk_repo = snapshot_download("MeiGen-AI/InfiniteTalk")
        self.infinitetalk_dir = os.path.join(infinitetalk_repo, "single/infinitetalk.safetensors")
        
        # Check if CausVid LoRA should be used based on environment variable
        use_causvid = os.environ.get("USE_CAUSVID_LORA", "false").lower() == "true"
        if use_causvid:
            # Download CausVid LoRA for enhanced quality
            causvid_repo = snapshot_download("MeiGen-AI/CausVid")
            self.lora_path = os.path.join(causvid_repo, "lora_v1.safetensors")
            logging.info(f"Using CausVid LoRA from {self.lora_path}")
        else:
            # No LoRA for standard variant
            self.lora_path = None
            logging.info("Using base model without LoRA")
        
        logging.info("Model downloads completed.")
        self.audio_save_dir = 'save_audio/inference'
        
        # Model configuration
        # Model config parameters - simplified for image-to-video only
        self.task = "infinitetalk-14B"
        self.task_mode = "SingleImageDriven"  # Fixed to image-to-video only
        self.default_frame_num = 125  # Default frame count for image-to-video (5 seconds)
        self.motion_frame = 0  # Starting motion frame
        self.mode = "clip"     # Mode for image-to-video
        self.sample_shift = 7  # Default for 480p
        # Disable offloading for high VRAM setups - check if GPU has enough VRAM
        gpu_vram_gb = 0
        if self.device.type == 'cuda' and torch.cuda.is_available():
            gpu_vram_gb = torch.cuda.get_device_properties(self.device).total_memory / (1024**3)
        # Only offload if less than 32GB VRAM or multi-GPU setup
        self.offload_model = False if gpu_vram_gb >= 32 else (True if self.world_size == 1 else False)
        self.color_correction_strength = 1.0
        
        logging.info(f"GPU VRAM: {gpu_vram_gb:.1f}GB, Model offloading: {self.offload_model}")
        
        # Initialize wav2vec components directly on GPU for faster loading
        os.makedirs(self.audio_save_dir, exist_ok=True)
        logging.info(f"Initializing Wav2Vec2 model directly on {self.device}")
        self.wav2vec_feature_extractor, self.audio_encoder = self._custom_init(self.device, self.wav2vec_dir)
        
        # Initialize WAN pipeline  
        cfg = WAN_CONFIGS[self.task]
        logging.info("Creating InfiniteTalk pipeline.")
            
        self.wan_pipeline = wan.InfiniteTalkPipeline(
            config=cfg,
            checkpoint_dir=self.ckpt_dir,
            quant_dir=None,  # Add quantization dir (None = no quantization)
            device_id=self.device_id,
            rank=self.rank,
            t5_fsdp=False,
            dit_fsdp=False,
            use_usp=False,
            t5_cpu=False,  # Keep T5 on GPU for high VRAM setups
            lora_dir=self.lora_path,   # Use CausVid LoRA if enabled
            lora_scales=None,  # No LoRA scales
            quant=None,      # Add quantization type (None = no quantization)
            dit_path=None,   # Add dit path
            infinitetalk_dir=self.infinitetalk_dir
        )
        
        logging.info("InfiniteTalk pipeline initialized successfully.")
        
        # Ensure flash_attn is being used for all attention mechanisms
        # Set flash_attention_2 on the T5 model if available
        if hasattr(self.wan_pipeline, 't5') and hasattr(self.wan_pipeline.t5, 'config'):
            self.wan_pipeline.t5.config.attn_implementation = 'flash_attention_2'
            logging.info("Set T5 attention implementation to flash_attention_2")
            
        # Log what attention implementation is actually being used
        try:
            import flash_attn
            logging.info(f"Flash Attention version: {flash_attn.__version__}")
        except ImportError:
            logging.warning("Flash Attention not available - this may cause issues!")
            
        # Create extra_args class for TeaCache and APG configuration
        class ExtraArgs:
            def __init__(self, use_teacache=False, teacache_thresh=0.1, size="infinitetalk-480"):
                self.use_teacache = use_teacache
                self.teacache_thresh = teacache_thresh
                self.size = size
                self.use_apg = False  # Disable APG by default
                self.apg_momentum = 0.9
                self.apg_norm_threshold = 0.1
                
        self.ExtraArgs = ExtraArgs  # Store class for later use
    
    def _custom_init(self, device, wav2vec_dir):
        """Initialize wav2vec components."""
        # Load with flash_attention_2 and proper dtype for flash attention compatibility
        audio_encoder = Wav2Vec2Model.from_pretrained(
            wav2vec_dir, 
            local_files_only=True,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16  # Flash attention requires fp16 or bf16
        ).to(device)
        logging.info("Wav2Vec2 using flash_attention_2 with bfloat16")
            
        audio_encoder.feature_extractor._freeze_parameters()
        wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_dir, local_files_only=True)
        return wav2vec_feature_extractor, audio_encoder
    
    def _loudness_norm(self, audio_array, sr=16000, lufs=-23):
        """Normalize audio loudness."""
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(audio_array)
        if abs(loudness) > 100:
            return audio_array
        normalized_audio = pyln.normalize.loudness(audio_array, loudness, lufs)
        return normalized_audio
    
    def _audio_prepare_single(self, audio_path, sample_rate=16000):
        """Prepare single audio file."""
        ext = os.path.splitext(audio_path)[1].lower()
        if ext in ['.mp4', '.mov', '.avi', '.mkv']:
            # Extract audio from video
            raw_audio_path = f"/tmp/{os.path.basename(audio_path).split('.')[0]}.wav"
            import subprocess
            ffmpeg_command = [
                "ffmpeg", "-y", "-i", str(audio_path), "-vn", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "2", str(raw_audio_path)
            ]
            subprocess.run(ffmpeg_command, check=True)
            human_speech_array, sr = librosa.load(raw_audio_path, sr=sample_rate)
            human_speech_array = self._loudness_norm(human_speech_array, sr)
            os.remove(raw_audio_path)
        else:
            human_speech_array, sr = librosa.load(audio_path, sr=sample_rate)
            human_speech_array = self._loudness_norm(human_speech_array, sr)
        return human_speech_array
    
    def _audio_prepare_multi(self, left_path, right_path, audio_type, sample_rate=16000):
        """Prepare multi-person audio files."""
        if not (left_path == 'None' or right_path == 'None'):
            human_speech_array1 = self._audio_prepare_single(left_path)
            human_speech_array2 = self._audio_prepare_single(right_path)
        elif left_path == 'None':
            human_speech_array2 = self._audio_prepare_single(right_path)
            human_speech_array1 = np.zeros(human_speech_array2.shape[0])
        elif right_path == 'None':
            human_speech_array1 = self._audio_prepare_single(left_path)
            human_speech_array2 = np.zeros(human_speech_array1.shape[0])

        if audio_type == 'para':
            new_human_speech1 = human_speech_array1
            new_human_speech2 = human_speech_array2
        elif audio_type == 'add':
            new_human_speech1 = np.concatenate([human_speech_array1[: human_speech_array1.shape[0]], np.zeros(human_speech_array2.shape[0])]) 
            new_human_speech2 = np.concatenate([np.zeros(human_speech_array1.shape[0]), human_speech_array2[:human_speech_array2.shape[0]]])
        
        sum_human_speechs = new_human_speech1 + new_human_speech2
        return new_human_speech1, new_human_speech2, sum_human_speechs
    
    def _get_embedding(self, speech_array, sr=16000, target_frames=None):
        """Extract audio embedding using wav2vec."""
        audio_duration = len(speech_array) / sr
        video_length = audio_duration * 25  # 25 fps
        
        # If target_frames is provided, ensure audio matches the target length
        if target_frames is not None:
            required_audio_length = target_frames / 25  # Convert frames to seconds
            if audio_duration < required_audio_length:
                # Pad audio with silence to match required length
                required_samples = int(required_audio_length * sr)
                padding_samples = required_samples - len(speech_array)
                speech_array = np.pad(speech_array, (0, padding_samples), mode='constant', constant_values=0)
                audio_duration = required_audio_length
                video_length = target_frames
                logging.info(f"Audio padded from {len(speech_array) - padding_samples} to {len(speech_array)} samples ({audio_duration:.2f}s) to match target {target_frames} frames")
            elif audio_duration > required_audio_length:
                # Log if audio is longer than target
                logging.info(f"Audio duration ({audio_duration:.2f}s) is longer than target ({required_audio_length:.2f}s). Video will be generated for full audio length.")
                # Update target_frames to match actual audio length
                target_frames = int(video_length)
        
        # Process with wav2vec - optimize for GPU processing
        audio_feature = np.squeeze(
            self.wav2vec_feature_extractor(speech_array, sampling_rate=sr).input_values
        )
        # Use bfloat16 and move directly to GPU in one operation
        audio_feature = torch.from_numpy(audio_feature).to(dtype=torch.bfloat16, device=self.device, non_blocking=True)
        audio_feature = audio_feature.unsqueeze(0)
        
        # Extract embeddings with optimized settings
        with torch.no_grad():
            # Use torch.cuda.amp for potential speedup and memory efficiency
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == 'cuda'):
                embeddings = self.audio_encoder(audio_feature, seq_len=int(video_length), output_hidden_states=True)
        
        if len(embeddings) == 0:
            raise RuntimeError("Failed to extract audio embedding")
        
        audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
        audio_emb = rearrange(audio_emb, "b s d -> s b d")
        
        # Keep on GPU and only transfer to CPU when actually needed for saving
        return audio_emb.detach(), int(video_length)
    
    def _process_tts_single(self, text, save_dir, voice1):
        """Process text-to-speech for single person."""
        pipeline = KPipeline(lang_code='a', repo_id=self.kokoro_dir)
        # Use default voice if custom voice path doesn't exist
        if not os.path.exists(voice1):
            voice_path = os.path.join(self.kokoro_dir, 'voices', 'af_sarah.pt')
        else:
            voice_path = voice1
        voice_tensor = torch.load(voice_path, weights_only=True)
        generator = pipeline(text, voice=voice_tensor, speed=1, split_pattern=r'\n+')
        
        audios = []
        for i, (gs, ps, audio) in enumerate(generator):
            audios.append(audio)
        audios = torch.concat(audios, dim=0)
        
        save_path1 = f'{save_dir}/s1.wav'
        sf.write(save_path1, audios, 24000)
        s1, _ = librosa.load(save_path1, sr=16000)
        return s1, save_path1

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Generate video using InfiniteTalk."""
        # Create extra_args with user input
        extra_args = self.ExtraArgs(
            use_teacache=input_data.use_teacache,
            teacache_thresh=input_data.teacache_thresh,
            size=input_data.resolution
        )
        
        # Prepare input data structure for image-to-video
        input_dict = {
            "prompt": input_data.prompt,
            "cond_video": input_data.image.path  # Use image as input
        }
        
        # Process single audio file
        human_speech = self._audio_prepare_single(input_data.audio_file.path)
        
        # Calculate frame count based on audio length
        audio_duration = len(human_speech) / 16000  # audio sample rate
        video_frames = max(25, int(audio_duration * 25))  # 25fps, minimum 25 frames (1 second)
        
        # The audio embedding needs to be longer than the video to allow windowed processing
        # Add some buffer frames for the processing pipeline
        embedding_frames = video_frames + 10  # Add buffer frames
        
        logging.info(f"Audio duration: {audio_duration:.2f}s, video frames: {video_frames} ({video_frames/25:.2f}s video)")
        
        audio_embedding, actual_embedding_frames = self._get_embedding(human_speech, target_frames=embedding_frames)
        
        # Set frame_num to the desired video length (not the embedding length)
        self.frame_num = video_frames
        
        # Save audio embeddings and processed audio
        emb_path = os.path.join(self.audio_save_dir, '1.pt')
        sum_audio = os.path.join(self.audio_save_dir, 'sum.wav')
        sf.write(sum_audio, human_speech, 16000)
        # Move to CPU only when saving to disk
        torch.save(audio_embedding.cpu(), emb_path)
        
        logging.info(f"Saved audio embedding shape: {audio_embedding.shape} to {emb_path}")
        logging.info(f"Video frame count: {self.frame_num}, audio embedding frames: {audio_embedding.shape[0]} (buffer: {audio_embedding.shape[0] - self.frame_num})")
        
        # Set audio paths in input dict
        input_dict['cond_audio'] = {'person1': emb_path}
        input_dict['video_audio'] = sum_audio
        
        # Generate video
        logging.info("Generating video...")
        video = self.wan_pipeline.generate_infinitetalk(
            input_dict,
            size_buckget=input_data.resolution,
            motion_frame=self.motion_frame,
            frame_num=self.frame_num,
            shift=self.sample_shift,
            sampling_steps=input_data.sampling_steps,
            text_guide_scale=input_data.text_guide_scale,
            audio_guide_scale=input_data.audio_guide_scale,
            seed=input_data.seed,
            n_prompt=input_data.negative_prompt,
            offload_model=self.offload_model,
            max_frames_num=self.frame_num if self.mode == 'clip' else 1000,
            color_correction_strength=self.color_correction_strength,
            extra_args=extra_args,  # Pass the dynamic extra_args
        )
        
        # Save video
        formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"infinitetalk_{input_data.resolution}_{formatted_time}"
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_output_path = temp_file.name
        
        save_video_ffmpeg(video, output_filename, [input_dict['video_audio']], high_quality_save=False)
        final_output_path = f"{output_filename}.mp4"
        
        # Move to temp location for return
        import shutil
        shutil.move(final_output_path, temp_output_path)
        
        logging.info(f"Video generated successfully: {temp_output_path}")
        
        return AppOutput(video=File(path=temp_output_path))

    async def unload(self):
        """Clean up resources."""
        if hasattr(self, 'wan_pipeline'):
            del self.wan_pipeline
        if hasattr(self, 'audio_encoder'):
            del self.audio_encoder
        if hasattr(self, 'wav2vec_feature_extractor'):
            del self.wav2vec_feature_extractor
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
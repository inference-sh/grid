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
from .InfiniteTalk.kokoro import KPipeline
from .InfiniteTalk.src.audio_analysis.wav2vec2 import Wav2Vec2Model

warnings.filterwarnings('ignore')

class AppInput(BaseAppInput):
    image: Optional[File] = Field(None, description="Input image for single image driven mode")
    video: Optional[File] = Field(None, description="Input video for video dubbing mode")
    prompt: str = Field(description="Text prompt describing the desired video")
    negative_prompt: str = Field(
        default="bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
        description="Negative prompt"
    )
    task_mode: str = Field(
        default="VideoDubbing", 
        description="Task mode", 
        enum=["SingleImageDriven", "VideoDubbing"]
    )
    audio_mode: str = Field(
        default="Single Person(Local File)",
        description="Audio mode",
        enum=["Single Person(Local File)", "Single Person(TTS)", "Multi Person(Local File, audio add)", "Multi Person(Local File, audio parallel)", "Multi Person(TTS)"]
    )
    audio_1: Optional[File] = Field(None, description="Audio file for speaker 1")
    audio_2: Optional[File] = Field(None, description="Audio file for speaker 2")
    tts_text: Optional[str] = Field(None, description="Text for TTS generation")
    resolution: str = Field(
        default="infinitetalk-480",
        description="Output resolution",
        enum=["infinitetalk-480", "infinitetalk-720"]
    )
    sampling_steps: int = Field(default=8, description="Number of sampling steps")
    seed: int = Field(default=42, description="Random seed")
    text_guide_scale: float = Field(default=1.0, description="Text guidance scale")
    audio_guide_scale: float = Field(default=2.0, description="Audio guidance scale")
    human1_voice: str = Field(
        default="weights/Kokoro-82M/voices/am_adam.pt",
        description="Voice file for person 1"
    )
    human2_voice: str = Field(
        default="weights/Kokoro-82M/voices/af_heart.pt", 
        description="Voice file for person 2"
    )

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
        
        # Download Kokoro TTS model for voices
        self.kokoro_dir = snapshot_download("hexgrad/Kokoro-82M")
        
        logging.info("Model downloads completed.")
        self.audio_save_dir = 'save_audio/inference'
        
        # Model configuration
        self.task = "infinitetalk-14B"
        self.frame_num = 81
        self.motion_frame = 9
        self.mode = "streaming"
        self.sample_shift = 7  # Default for 480p
        self.offload_model = True if self.world_size == 1 else False
        self.color_correction_strength = 1.0
        
        # Initialize wav2vec components on CPU first (like original app.py), then move to device
        os.makedirs(self.audio_save_dir, exist_ok=True)
        self.wav2vec_feature_extractor, self.audio_encoder = self._custom_init('cpu', self.wav2vec_dir)
        # Now move to actual device
        self.audio_encoder = self.audio_encoder.to(self.device)
        
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
            t5_cpu=(self.device.type != 'cuda'),
            lora_dir=None,   # Add lora dir (None = no LoRA)
            lora_scales=None,  # Add lora scales
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
            
        # Create extra_args object for TeaCache and APG configuration
        class ExtraArgs:
            def __init__(self):
                self.use_teacache = False  # Disable TeaCache by default
                self.teacache_thresh = 0.1
                self.size = "infinitetalk-480"
                self.use_apg = False  # Disable APG by default
                self.apg_momentum = 0.9
                self.apg_norm_threshold = 0.1
                
        self.extra_args = ExtraArgs()
    
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
    
    def _get_embedding(self, speech_array, sr=16000):
        """Extract audio embedding using wav2vec."""
        audio_duration = len(speech_array) / sr
        video_length = audio_duration * 25  # 25 fps
        
        # Process with wav2vec
        audio_feature = np.squeeze(
            self.wav2vec_feature_extractor(speech_array, sampling_rate=sr).input_values
        )
        # Use bfloat16 to match model weights
        audio_feature = torch.from_numpy(audio_feature).to(dtype=torch.bfloat16, device=self.device)
        audio_feature = audio_feature.unsqueeze(0)
        
        # Extract embeddings
        with torch.no_grad():
            embeddings = self.audio_encoder(audio_feature, seq_len=int(video_length), output_hidden_states=True)
        
        if len(embeddings) == 0:
            raise RuntimeError("Failed to extract audio embedding")
        
        audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
        audio_emb = rearrange(audio_emb, "b s d -> s b d")
        
        return audio_emb.cpu().detach()
    
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
        # Prepare input data structure
        input_dict = {
            "prompt": input_data.prompt
        }
        
        # Set condition video/image based on task mode
        if input_data.task_mode == 'VideoDubbing':
            if input_data.video is None:
                raise ValueError("Video is required for VideoDubbing mode")
            input_dict["cond_video"] = input_data.video.path
        else:  # SingleImageDriven
            if input_data.image is None:
                raise ValueError("Image is required for SingleImageDriven mode")
            input_dict["cond_video"] = input_data.image.path
        
        # Process audio based on mode
        person = {}
        if input_data.audio_mode == "Single Person(Local File)":
            if input_data.audio_1 is None:
                raise ValueError("Audio file is required for Single Person mode")
            person['person1'] = input_data.audio_1.path
        elif input_data.audio_mode == "Single Person(TTS)":
            if input_data.tts_text is None:
                raise ValueError("TTS text is required for TTS mode")
            tts_audio = {
                'text': input_data.tts_text,
                'human1_voice': input_data.human1_voice
            }
            input_dict["tts_audio"] = tts_audio
        # Add other audio modes as needed...
        
        input_dict["cond_audio"] = person
        
        # Process audio files and generate embeddings
        if 'Local File' in input_data.audio_mode and len(person) >= 1:
            human_speech = self._audio_prepare_single(person['person1'])
            audio_embedding = self._get_embedding(human_speech)
            emb_path = os.path.join(self.audio_save_dir, '1.pt')
            sum_audio = os.path.join(self.audio_save_dir, 'sum.wav')
            sf.write(sum_audio, human_speech, 16000)
            torch.save(audio_embedding, emb_path)
            input_dict['cond_audio']['person1'] = emb_path
            input_dict['video_audio'] = sum_audio
        elif 'TTS' in input_data.audio_mode:
            if 'human2_voice' not in input_dict['tts_audio']:
                new_human_speech1, sum_audio = self._process_tts_single(
                    input_dict['tts_audio']['text'], 
                    self.audio_save_dir, 
                    input_dict['tts_audio']['human1_voice']
                )
                audio_embedding_1 = self._get_embedding(new_human_speech1)
                emb1_path = os.path.join(self.audio_save_dir, '1.pt')
                torch.save(audio_embedding_1, emb1_path)
                input_dict['cond_audio']['person1'] = emb1_path
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
            extra_args=self.extra_args,  # Pass the extra_args to fix use_teacache error
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
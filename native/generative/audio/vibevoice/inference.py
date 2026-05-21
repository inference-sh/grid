import os
import sys
import tempfile
import time
import numpy as np
import soundfile as sf
import librosa
import torch
from typing import List, Optional
from pathlib import Path
from enum import Enum

# Add current directory to Python path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "VibeVoice"))

# Enable faster HF downloads globally
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from accelerate import Accelerator
from huggingface_hub import snapshot_download
from transformers.utils import logging
from transformers import set_seed

logging.set_verbosity_info()
logger = logging.get_logger(__name__)


class VoicePreset(str, Enum):
    """Available voice presets for speakers."""
    ALICE_WOMAN = "en-Alice_woman"
    CARTER_MAN = "en-Carter_man"
    FRANK_MAN = "en-Frank_man"
    MARY_WOMAN_BGM = "en-Mary_woman_bgm"
    MAYA_WOMAN = "en-Maya_woman"
    SAMUEL_MAN = "in-Samuel_man"
    ANCHEN_MAN_BGM = "zh-Anchen_man_bgm"
    BOWEN_MAN = "zh-Bowen_man"
    XINRAN_WOMAN = "zh-Xinran_woman"


class AppInput(BaseAppInput):
    script: str = Field(
        description="The conversation script text. Can be plain text (auto-assigned to speakers) or formatted with 'Speaker X:' prefixes"
    )
    num_speakers: int = Field(
        description="Number of speakers (1-4)", 
        ge=1, 
        le=4, 
        default=1
    )
    voice_presets: Optional[List[VoicePreset]] = Field(
        default=None,
        description="Voice presets for each speaker. If not provided, default voices will be automatically selected.",
        max_items=4
    )
    custom_voice_samples: Optional[List[File]] = Field(
        default=None,
        description="Custom voice sample audio files (WAV format, one per speaker). If provided, these override voice_presets.",
        max_items=4
    )
    cfg_scale: float = Field(
        description="CFG Scale (guidance strength)", 
        ge=1.0, 
        le=2.0, 
        default=1.3
    )
    inference_steps: int = Field(
        description="Number of diffusion inference steps", 
        ge=3, 
        le=20, 
        default=10
    )


class AppOutput(BaseAppOutput):
    audio: File = Field(description="Generated podcast audio file")
    duration: float = Field(description="Audio duration in seconds")
    generation_time: float = Field(description="Time taken to generate audio in seconds")
    rtf: float = Field(description="Real Time Factor (generation_time / audio_duration)")
    input_tokens: int = Field(description="Number of input tokens")
    generated_tokens: int = Field(description="Number of generated tokens")
    model_status: str = Field(description="Status of model availability")


class App(BaseApp):
    def __init__(self):
        super().__init__()
        self.model = None
        self.processor = None
        self.device = None
        self.accelerator = None
        self.voice_presets_dict = {}
        self.model_available = False

    def check_flash_attention_availability(self):
        """Check if flash_attn is available"""
        try:
            import flash_attn
            return True
        except ImportError:
            return False

    async def setup(self, metadata):
        """Initialize the VibeVoice model based on variant."""
        # Get variant from metadata
        variant = getattr(metadata, "app_variant", "default")
        
        # Check flash attention availability
        flash_attn_available = self.check_flash_attention_availability()
        
        logger.info("="*60)
        logger.info("VIBEVOICE MODEL SETUP")
        logger.info("="*60)
        logger.info(f"Variant: {variant}")
        logger.info(f"Flash Attention Available: {flash_attn_available}")
        logger.info(f"Available variants: default, vibevoice-1.5b-lowmem, vibevoice-7b, vibevoice-7b-lowmem")
        
        # Map variants to actual model configurations
        if variant in ["default"]:
            model_id = "microsoft/VibeVoice-1.5b"
            inference_steps = 10
            low_cpu_mem_usage = False  # Fast load
            model_name = "VibeVoice 1.5B"
        elif variant == "vibevoice-1.5b-lowmem":
            model_id = "microsoft/VibeVoice-1.5b"
            inference_steps = 10
            low_cpu_mem_usage = True  # Low memory
            model_name = "VibeVoice 1.5B (Low Memory)"
        elif variant == "vibevoice-7b":
            model_id = "microsoft/VibeVoice-Large"
            inference_steps = 10
            low_cpu_mem_usage = False  # Fast load
            model_name = "VibeVoice 7B"
        elif variant == "vibevoice-7b-lowmem":
            model_id = "microsoft/VibeVoice-Large"
            inference_steps = 10
            low_cpu_mem_usage = True  # Low memory
            model_name = "VibeVoice 7B (Low Memory)"
        else:
            # Default to 1.5b
            logger.warning(f"Unknown variant '{variant}', defaulting to 1.5B")
            model_id = "microsoft/VibeVoice-1.5b"
            inference_steps = 10
            low_cpu_mem_usage = False
            model_name = "VibeVoice 1.5B (Default)"
        
        logger.info(f"Model ID: {model_id}")
        logger.info(f"Model Name: {model_name}")
        logger.info(f"Low CPU Memory Usage: {low_cpu_mem_usage}")
        logger.info(f"Default Inference Steps: {inference_steps}")
        
        self.variant = variant
        self.model_name = model_name
        self.model_id = model_id
        self.default_inference_steps = inference_steps
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self.flash_attn_available = flash_attn_available
        
        # Setup device with accelerate
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        
        logger.info(f"Device: {self.device}")
        
        # Setup voice presets
        self.setup_voice_presets()
        
        # Try to load VibeVoice
        try:
            # Import VibeVoice components
            from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
            from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
            
            # Download model to HuggingFace cache
            logger.info(f"Downloading model: {model_id}")
            try:
                self.model_path = snapshot_download(
                    repo_id=model_id,
                    resume_download=True,
                    local_files_only=False,
                )
                logger.info(f"✓ Model downloaded to: {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to download model from HF: {e}")
                # Try local path
                self.model_path = "./VibeVoice"
                if not os.path.exists(self.model_path):
                    raise ValueError(f"Model not found locally at {self.model_path}")
            
            # Load processor
            logger.info("Loading processor...")
            self.processor = VibeVoiceProcessor.from_pretrained(self.model_path)
            logger.info("✓ Processor loaded")
            
            # Load model with attention detection
            logger.info(f"Loading {model_name}...")
            torch_dtype = torch.bfloat16
            device_map = 'cuda'  # Same as demo
            
            # Determine attention implementation
            if flash_attn_available:
                logger.info("🚀 Flash Attention 2 is available - attempting to use it")
                try:
                    self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                        self.model_path,
                        torch_dtype=torch_dtype,
                        device_map=device_map,
                        attn_implementation="flash_attention_2",
                        low_cpu_mem_usage=low_cpu_mem_usage,
                    )
                    self.attention_type = "flash_attention_2"
                    logger.info("✓ Successfully loaded with Flash Attention 2")
                except Exception as flash_error:
                    logger.warning(f"Flash Attention 2 failed: {flash_error}")
                    logger.info("Falling back to SDPA attention")
                    self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                        self.model_path,
                        torch_dtype=torch_dtype,
                        device_map=device_map,
                        attn_implementation="sdpa",
                        low_cpu_mem_usage=low_cpu_mem_usage,
                    )
                    self.attention_type = "sdpa"
                    logger.info("✓ Successfully loaded with SDPA attention")
            else:
                logger.info("Flash Attention 2 not available - using SDPA")
                try:
                    self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                        self.model_path,
                        torch_dtype=torch_dtype,
                        device_map=device_map,
                        attn_implementation="sdpa",
                        low_cpu_mem_usage=low_cpu_mem_usage,
                    )
                    self.attention_type = "sdpa"
                    logger.info("✓ Successfully loaded with SDPA attention")
                except Exception as sdpa_error:
                    logger.warning(f"SDPA attention failed: {sdpa_error}")
                    logger.info("Using default attention implementation")
                    self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                        self.model_path,
                        torch_dtype=torch_dtype,
                        device_map=device_map,
                        low_cpu_mem_usage=low_cpu_mem_usage,
                    )
                    self.attention_type = "default"
                    logger.info("✓ Successfully loaded with default attention")
            
            self.model.eval()
            
            # Set default inference steps
            self.model.set_ddpm_inference_steps(num_steps=self.default_inference_steps)
            
            # Print final configuration
            logger.info("="*60)
            logger.info("MODEL LOADED SUCCESSFULLY")
            logger.info("="*60)
            logger.info(f"✓ Model: {self.model_name}")
            logger.info(f"✓ Model ID: {self.model_id}")
            logger.info(f"✓ Attention Type: {self.attention_type}")
            logger.info(f"✓ Low CPU Memory Usage: {self.low_cpu_mem_usage}")
            logger.info(f"✓ Default Inference Steps: {self.default_inference_steps}")
            logger.info(f"✓ Device: {self.device}")
            logger.info(f"✓ Torch dtype: {torch_dtype}")
            
            if hasattr(self.model.model, 'language_model'):
                lm_attn = self.model.model.language_model.config._attn_implementation
                logger.info(f"✓ Language model attention: {lm_attn}")
            
            self.model_available = True
            logger.info("="*60)
            
        except Exception as e:
            logger.warning(f"VibeVoice model not available: {e}")
            logger.info("Running in demo mode - will generate placeholder audio")
            self.model_available = False
            self.model = None
            self.processor = None
        
        set_seed(42)  # Set seed for reproducibility

    def setup_voice_presets(self):
        """Setup voice presets by scanning the voices directory."""
        voices_dir = "./voices"
        
        if not os.path.exists(voices_dir):
            logger.warning(f"Voices directory not found at {voices_dir}")
            self.voice_presets_dict = {}
            return
        
        # Get all .wav files in the voices directory
        wav_files = [f for f in os.listdir(voices_dir) 
                    if f.lower().endswith('.wav') and os.path.isfile(os.path.join(voices_dir, f))]
        
        # Create dictionary with filename (without extension) as key
        for wav_file in wav_files:
            # Remove .wav extension to get the name
            name = os.path.splitext(wav_file)[0]
            # Create full path
            full_path = os.path.join(voices_dir, wav_file)
            self.voice_presets_dict[name] = full_path
        
        # Sort the voice presets alphabetically by name
        self.voice_presets_dict = dict(sorted(self.voice_presets_dict.items()))
        
        logger.info(f"Found {len(self.voice_presets_dict)} voice files in {voices_dir}")
        logger.info(f"Available voices: {', '.join(self.voice_presets_dict.keys())}")

    def read_audio(self, audio_path: str, target_sr: int = 24000) -> np.ndarray:
        """Read and preprocess audio file."""
        try:
            wav, sr = sf.read(audio_path)
            if len(wav.shape) > 1:
                wav = np.mean(wav, axis=1)  # Convert to mono
            if sr != target_sr:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
            return wav.astype(np.float32)
        except Exception as e:
            raise ValueError(f"Error reading audio {audio_path}: {e}")

    def format_script(self, script: str, num_speakers: int) -> str:
        """Format script with speaker assignments."""
        lines = script.strip().split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line already has speaker format
            if line.startswith('Speaker ') and ':' in line:
                formatted_lines.append(line)
            else:
                # Auto-assign to speakers in rotation
                speaker_id = len(formatted_lines) % num_speakers
                formatted_lines.append(f"Speaker {speaker_id}: {line}")
        
        return '\n'.join(formatted_lines)

    def get_voice_files(self, input_data: AppInput) -> List[str]:
        """Get voice file paths based on input preferences."""
        voice_files = []
        
        # Priority: custom_voice_samples > voice_presets > defaults
        if input_data.custom_voice_samples and len(input_data.custom_voice_samples) > 0:
            # Use custom voice samples (highest priority)
            if len(input_data.custom_voice_samples) != input_data.num_speakers:
                raise ValueError(f"Number of custom voice samples ({len(input_data.custom_voice_samples)}) must match number of speakers ({input_data.num_speakers})")
            
            for voice_file in input_data.custom_voice_samples:
                if not voice_file.exists():
                    raise ValueError(f"Custom voice sample not found: {voice_file.path}")
                voice_files.append(voice_file.path)
            
            logger.info("Using custom voice samples")
            
        elif input_data.voice_presets and len(input_data.voice_presets) > 0:
            # Use voice presets (medium priority)
            if len(input_data.voice_presets) != input_data.num_speakers:
                raise ValueError(f"Number of voice presets ({len(input_data.voice_presets)}) must match number of speakers ({input_data.num_speakers})")
            
            for preset in input_data.voice_presets:
                preset_name = preset.value
                if preset_name not in self.voice_presets_dict:
                    raise ValueError(f"Voice preset not found: {preset_name}")
                voice_files.append(self.voice_presets_dict[preset_name])
            
            logger.info(f"Using voice presets: {[preset.value for preset in input_data.voice_presets]}")
            
        else:
            # Use default voices (lowest priority)
            default_presets = [
                VoicePreset.ALICE_WOMAN,
                VoicePreset.CARTER_MAN,
                VoicePreset.MAYA_WOMAN,
                VoicePreset.FRANK_MAN
            ]
            
            for i in range(input_data.num_speakers):
                preset = default_presets[i % len(default_presets)]
                preset_name = preset.value
                if preset_name not in self.voice_presets_dict:
                    raise ValueError(f"Default voice preset not found: {preset_name}")
                voice_files.append(self.voice_presets_dict[preset_name])
            
            used_presets = [default_presets[i % len(default_presets)].value for i in range(input_data.num_speakers)]
            logger.info(f"Using default voice presets: {used_presets}")
        
        return voice_files

    def generate_demo_audio(self, input_data: AppInput, voice_files: List[str]) -> tuple:
        """Generate demo audio when VibeVoice model is not available."""
        sample_rate = 24000
        # Estimate duration based on script length (rough approximation)
        estimated_duration = max(3.0, len(input_data.script.split()) * 0.6)  # ~0.6 seconds per word
        
        # Generate multiple sine waves to simulate different speakers
        t = np.linspace(0, estimated_duration, int(sample_rate * estimated_duration))
        audio_np = np.zeros_like(t)
        
        # Create distinct frequencies for different speakers
        for i in range(input_data.num_speakers):
            freq = 220 + i * 110 + (hash(input_data.script) % 50)  # Different freq per speaker
            speaker_audio = 0.2 * np.sin(2 * np.pi * freq * t)
            
            # Add speaker timing (simple alternation)
            start_time = i * (estimated_duration / input_data.num_speakers)
            end_time = (i + 1) * (estimated_duration / input_data.num_speakers)
            start_idx = int(start_time * sample_rate)
            end_idx = int(end_time * sample_rate)
            
            if end_idx <= len(audio_np):
                audio_np[start_idx:end_idx] += speaker_audio[start_idx:end_idx]
        
        # Add envelope for more natural sound
        envelope = np.exp(-2 * np.abs(t - estimated_duration/2) / estimated_duration)
        audio_np = audio_np * envelope
        
        # Normalize
        if np.max(np.abs(audio_np)) > 0:
            audio_np = audio_np / np.max(np.abs(audio_np)) * 0.8
        
        return audio_np, sample_rate, estimated_duration

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Generate podcast audio from script and voice samples."""
        start_time = time.time()
        
        # Validate inputs
        if not input_data.script.strip():
            raise ValueError("Script cannot be empty")
        
        # Get voice files based on input preferences
        voice_files = self.get_voice_files(input_data)
        
        # Read voice samples for validation
        voice_samples = []
        for voice_file in voice_files:
            audio_data = self.read_audio(voice_file)
            voice_samples.append(audio_data)
        
        # Format script with speaker assignments
        formatted_script = self.format_script(input_data.script, input_data.num_speakers)
        formatted_script = formatted_script.replace("'", "'")  # Fix quote characters
        
        logger.info(f"Formatted script preview: {formatted_script[:200]}...")
        logger.info(f"Voice files: {[os.path.basename(f) for f in voice_files]}")
        
        if self.model_available and self.model is not None and self.processor is not None:
            # Real VibeVoice generation
            logger.info("Using real VibeVoice model for generation")
            
            # Set inference steps for this generation
            steps = max(input_data.inference_steps, 3)  # Minimum 3 steps
            self.model.set_ddpm_inference_steps(num_steps=steps)
            
            # Prepare inputs for the model
            inputs = self.processor(
                text=[formatted_script],  # Wrap in list for batch processing
                voice_samples=[voice_samples],  # Wrap in list for batch processing
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            
            logger.info(f"Starting generation with cfg_scale: {input_data.cfg_scale}, steps: {steps}")
            
            # Generate audio
            generation_start = time.time()
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=input_data.cfg_scale,
                tokenizer=self.processor.tokenizer,
                generation_config={'do_sample': False},
                verbose=True,
            )
            generation_time = time.time() - generation_start
            
            # Calculate metrics
            if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
                # Assuming 24kHz sample rate
                sample_rate = 24000
                audio_samples = outputs.speech_outputs[0].shape[-1] if len(outputs.speech_outputs[0].shape) > 0 else len(outputs.speech_outputs[0])
                audio_duration = audio_samples / sample_rate
                rtf = generation_time / audio_duration if audio_duration > 0 else float('inf')
            else:
                raise ValueError("No audio output generated")
            
            # Calculate token metrics
            input_tokens = inputs['input_ids'].shape[1]
            output_tokens = outputs.sequences.shape[1]
            generated_tokens = output_tokens - input_tokens
            
            # Save output
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                output_path = tmp_file.name
            
            self.processor.save_audio(
                outputs.speech_outputs[0],  # First (and only) batch item
                output_path=output_path,
            )
            
            model_status = "VibeVoice model active"
            
        else:
            # Demo mode - generate placeholder audio
            logger.info("Using demo mode (VibeVoice model not available)")
            
            generation_start = time.time()
            audio_np, sample_rate, audio_duration = self.generate_demo_audio(input_data, voice_files)
            generation_time = time.time() - generation_start
            
            rtf = generation_time / audio_duration if audio_duration > 0 else float('inf')
            
            # Mock token metrics for demo
            input_tokens = len(formatted_script.split()) * 2  # Rough approximation
            generated_tokens = input_tokens * 3  # Mock output tokens
            
            # Save output
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                output_path = tmp_file.name
            
            # Convert to 16-bit and save
            audio_16bit = (audio_np * 32767).astype(np.int16)
            sf.write(output_path, audio_16bit, sample_rate, subtype='PCM_16')
            
            model_status = "Demo mode - VibeVoice model not available"
        
        total_time = time.time() - start_time
        
        logger.info(f"Generated {audio_duration:.2f}s audio in {generation_time:.2f}s (RTF: {rtf:.2f}x)")
        logger.info(f"Tokens - Input: {input_tokens}, Generated: {generated_tokens}")
        
        return AppOutput(
            audio=File(path=output_path),
            duration=audio_duration,
            generation_time=generation_time,
            rtf=rtf,
            input_tokens=input_tokens,
            generated_tokens=generated_tokens,
            model_status=model_status
        )

    async def unload(self):
        """Clean up resources."""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        if hasattr(self, 'processor') and self.processor is not None:
            del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
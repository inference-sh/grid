import os
# Enable HF Hub fast transfer for faster model downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import numpy as np
import torch
import gc
import tempfile
import math
import logging
import shutil
import subprocess
import requests
from typing import Optional
from pydantic import Field
from PIL import Image
from diffusers import AutoencoderKLWan, WanSpeechToVideoPipeline, WanS2VTransformer3DModel, GGUFQuantizationConfig
from diffusers.utils import export_to_video, load_image, load_audio
from diffusers.utils.constants import DIFFUSERS_REQUEST_TIMEOUT
from diffusers.hooks import apply_first_block_cache, FirstBlockCacheConfig
from huggingface_hub import hf_hub_download
from accelerate import Accelerator
from transformers import Wav2Vec2ForCTC

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File

# Model variants mapping for GGUF quantization from QuantStack
# Using S2V model variants from QuantStack/Wan2.2-S2V-14B-GGUF
MODEL_VARIANTS = {
    "default": None,  # Use default F16 model
    "q2_k": "Wan2.2-S2V-14B-Q2_K.gguf",
    "q3_k_s": "Wan2.2-S2V-14B-Q3_K_S.gguf",
    "q3_k_m": "Wan2.2-S2V-14B-Q3_K_M.gguf",
    "q4_0": "Wan2.2-S2V-14B-Q4_0.gguf",
    "q4_1": "Wan2.2-S2V-14B-Q4_1.gguf",
    "q4_k_s": "Wan2.2-S2V-14B-Q4_K_S.gguf",
    "q4_k_m": "Wan2.2-S2V-14B-Q4_K_M.gguf",
    "q5_0": "Wan2.2-S2V-14B-Q5_0.gguf",
    "q5_1": "Wan2.2-S2V-14B-Q5_1.gguf",
    "q5_k_s": "Wan2.2-S2V-14B-Q5_K_S.gguf",
    "q5_k_m": "Wan2.2-S2V-14B-Q5_K_M.gguf",
    "q6_k": "Wan2.2-S2V-14B-Q6_K.gguf",
    "q8_0": "Wan2.2-S2V-14B-Q8_0.gguf"
}

DEFAULT_VARIANT = "default"


def get_size_less_than_area(height, width, target_area=1024 * 704, divisor=64):
    """Calculate optimal dimensions that fit within target area while maintaining aspect ratio."""
    if height * width <= target_area:
        max_upper_area = target_area
        min_scale = 0.1
        max_scale = 1.0
    else:
        max_upper_area = target_area
        d = divisor - 1
        b = d * (height + width)
        a = height * width
        c = d**2 - max_upper_area

        min_scale = (-b + math.sqrt(b**2 - 2 * a * c)) / (2 * a)
        max_scale = math.sqrt(max_upper_area / (height * width))

    find_it = False
    for i in range(100):
        scale = max_scale - (max_scale - min_scale) * i / 100
        new_height, new_width = int(height * scale), int(width * scale)

        pad_height = (64 - new_height % 64) % 64
        pad_width = (64 - new_width % 64) % 64
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        padded_height, padded_width = new_height + pad_height, new_width + pad_width

        if padded_height * padded_width <= max_upper_area:
            find_it = True
            break

    if find_it:
        return padded_height, padded_width
    else:
        aspect_ratio = width / height
        target_width = int((target_area * aspect_ratio)**0.5 // divisor * divisor)
        target_height = int((target_area / aspect_ratio)**0.5 // divisor * divisor)

        if target_width >= width or target_height >= height:
            target_width = int(width // divisor * divisor)
            target_height = int(height // divisor * divisor)

        return target_height, target_width

def aspect_ratio_resize(image, max_area=720 * 1280):
    """Resize image to fit within max area while maintaining aspect ratio."""
    height, width = get_size_less_than_area(image.size[1], image.size[0], target_area=max_area)
    image = image.resize((width, height))
    return image, height, width

def merge_video_audio(video_path: str, audio_path: str):
    """Merge video and audio into a new video, overwriting the original."""
    logging.basicConfig(level=logging.INFO)
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"video file {video_path} does not exist")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"audio file {audio_path} does not exist")

    base, ext = os.path.splitext(video_path)
    temp_output = f"{base}_temp{ext}"

    try:
        command = [
            'ffmpeg',
            '-y',  # overwrite
            '-i', video_path,
            '-i', audio_path,
            '-c:v', 'copy',  # copy video stream
            '-c:a', 'aac',   # use AAC audio encoder
            '-b:a', '192k',  # set audio bitrate
            '-map', '0:v:0', # select first video stream
            '-map', '1:a:0', # select first audio stream
            '-shortest',     # choose shortest duration
            temp_output
        ]

        logging.info("Start merging video and audio...")
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if result.returncode != 0:
            error_msg = f"FFmpeg execute failed: {result.stderr}"
            logging.error(error_msg)
            raise RuntimeError(error_msg)

        shutil.move(temp_output, video_path)
        logging.info(f"Merge completed, saved to {video_path}")

    except Exception as e:
        if os.path.exists(temp_output):
            os.remove(temp_output)
        logging.error(f"merge_video_audio failed with error: {e}")
        raise

class AppInput(BaseAppInput):
    image: File = Field(description="First frame image for video generation")
    audio: File = Field(description="Audio file for speech-to-video generation")
    prompt: str = Field(description="Text prompt describing the desired video content")
    resolution: str = Field(default="720p", description="Resolution preset", enum=["480p", "720p"])
    num_frames_per_chunk: int = Field(default=81, description="Number of frames to generate per chunk")
    fps: int = Field(default=16, description="Frames per second for the output video")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    enable_first_block_cache: bool = Field(default=True, description="Enable first block cache for faster inference")
    first_block_cache_threshold: float = Field(default=0.05, ge=0.01, le=0.5, description="Threshold for first block cache (lower = more acceleration, higher = better quality)")

class AppOutput(BaseAppOutput):
    file: File = Field(description="Generated video file with merged audio")



class App(BaseApp):
    async def setup(self, metadata):
        """Initialize the Wan2.2 Speech-to-Video pipeline."""
        print("Setting up Wan2.2 Speech-to-Video pipeline...")
        
        # Store resolution presets (same as wan2-2-i2v-a14b)
        self.resolution_presets = {
            "480p": {"max_area": 480 * 832},
            "720p": {"max_area": 720 * 1280}
        }
        
        # Initialize accelerator for proper device management
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        
        # Get variant and determine quantization/offloading strategy
        variant = getattr(metadata, "app_variant", DEFAULT_VARIANT)
        use_offload = variant.endswith("_offload")
        # Strip offload suffix for base variant lookup
        base_variant = variant.replace("_offload", "")
        if base_variant not in MODEL_VARIANTS:
            logging.warning(f"Unknown variant '{variant}', falling back to default '{DEFAULT_VARIANT}'")
            base_variant = DEFAULT_VARIANT
        
        # Model ID for the S2V model
        model_id = "tolgacangoz/Wan2.2-S2V-14B-Diffusers"
        
        # Load common components
        print("Loading audio encoder...")
        audio_encoder = Wav2Vec2ForCTC.from_pretrained(model_id, subfolder="audio_encoder", torch_dtype=torch.float32)
        
        print("Loading VAE...")
        vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
        
        # Load pipeline based on variant
        if base_variant == "default":
            # Load standard F16 pipeline
            print("Loading standard F16 pipeline...")
            self.pipe = WanSpeechToVideoPipeline.from_pretrained(
                model_id, vae=vae, audio_encoder=audio_encoder, torch_dtype=torch.bfloat16,
            )
        else:
            # Load quantized transformer
            print(f"Loading quantized transformer variant: {base_variant}")
            repo_id = "QuantStack/Wan2.2-S2V-14B-GGUF" 
            gguf_file = MODEL_VARIANTS[base_variant]
            
            # Download and load quantized transformer
            gguf_path = hf_hub_download(repo_id=repo_id, filename=gguf_file)
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            quantized_transformer = WanS2VTransformer3DModel.from_single_file(
                gguf_path,
                quantization_config=GGUFQuantizationConfig(compute_dtype=self.dtype),
                config=model_id,
                subfolder="transformer",
                torch_dtype=self.dtype,
            )
            
            # Load pipeline with quantized transformer
            print("Loading pipeline with quantized transformer...")
            self.pipe = WanSpeechToVideoPipeline.from_pretrained(
                model_id, 
                transformer=quantized_transformer,
                vae=vae, 
                audio_encoder=audio_encoder, 
                torch_dtype=torch.bfloat16,
            )
        
        # Apply offloading strategy
        if use_offload:
            print("Enabling CPU offload...")
            self.pipe.enable_model_cpu_offload()
        else:
            # Move pipeline to device
            self.pipe = self.pipe.to(self.device)
        
        # Set attention backend
        try:
            print("Setting flash attention backend...")
            self.pipe.transformer.set_attention_backend("flash")
        except Exception as e:
            print(f"Warning: Could not set flash attention: {e}")
        
        print(f"Setup complete! Using variant: {variant}, device: {self.device}")

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Generate video from image, audio, and text prompt."""
        print(f"Generating speech-to-video with prompt: {input_data.prompt}")
        
        # Apply first block cache if enabled
        if input_data.enable_first_block_cache:
            print(f"Applying first block cache with threshold: {input_data.first_block_cache_threshold}")
            try:
                config = FirstBlockCacheConfig(threshold=input_data.first_block_cache_threshold)
                apply_first_block_cache(self.pipe.transformer, config)
                print("First block cache applied successfully")
            except Exception as e:
                print(f"Warning: Could not apply first block cache: {e}")
        
        # Use resolution preset
        preset = self.resolution_presets.get(input_data.resolution, self.resolution_presets["720p"])
        max_area = preset["max_area"]
        print(f"Using resolution preset: {input_data.resolution} (max_area: {max_area})")

        # Load and process input image
        print("Loading and processing image...")
        if input_data.image.path.startswith('http'):
            first_frame = load_image(input_data.image.path)
        else:
            first_frame = load_image(input_data.image.path)
        
        # Resize image to fit within max area
        first_frame, height, width = aspect_ratio_resize(first_frame, max_area)
        print(f"Resized image to: {width}x{height}")
        
        # Load audio file
        print("Loading audio...")
        if input_data.audio.path.startswith('http'):
            audio, sampling_rate = load_audio(input_data.audio.path)
        else:
            audio, sampling_rate = load_audio(input_data.audio.path)
        print(f"Audio loaded: {audio.shape}, sampling rate: {sampling_rate}")
        
        # Set seed if provided
        generator = None
        if input_data.seed is not None:
            generator = torch.Generator().manual_seed(input_data.seed)
            print(f"Using seed: {input_data.seed}")
        
        # Generate video exactly like the example
        print("Starting video generation...")
        output = self.pipe(
            image=first_frame,
            audio=audio,
            sampling_rate=sampling_rate,
            prompt=input_data.prompt,
            height=height,
            width=width,
            num_frames_per_chunk=input_data.num_frames_per_chunk,
            generator=generator,
        ).frames[0]
        
        print("Video generation complete, exporting...")
        
        # Create temporary files for video output and audio merge
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
            video_path = temp_video.name
        
        # Export video without audio first
        export_to_video(output, video_path, fps=input_data.fps)
        print(f"Video exported to: {video_path}")
        
        # Download audio file if it's a URL for merging
        audio_path = input_data.audio.path
        if input_data.audio.path.startswith('http'):
            response = requests.get(input_data.audio.path, stream=True, timeout=DIFFUSERS_REQUEST_TIMEOUT)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
                for chunk in response.iter_content(chunk_size=8192):
                    temp_audio.write(chunk)
                audio_path = temp_audio.name
        
        # Merge original audio with the generated video
        print("Merging audio with video...")
        try:
            merge_video_audio(video_path, audio_path)
            print("Audio merge complete!")
        except Exception as e:
            print(f"Warning: Audio merge failed: {e}")
            print("Proceeding with video-only output...")
        
        # Clean up temporary audio file if it was downloaded
        if input_data.audio.path.startswith('http') and os.path.exists(audio_path):
            try:
                os.unlink(audio_path)
            except Exception:
                pass
        
        # Cleanup memory
        del output
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return AppOutput(file=File(path=video_path))

    async def unload(self):
        """Clean up resources."""
        print("Cleaning up...")
        if hasattr(self, 'pipe'):
            del self.pipe
        
        # Clear GPU cache if using CUDA
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("Cleanup complete!")
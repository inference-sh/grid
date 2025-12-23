import torch
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, UniPCMultistepScheduler
from diffusers.utils import export_to_video
from transformers import CLIPVisionModel
import tempfile
from huggingface_hub import hf_hub_download
import numpy as np
from PIL import Image
import random
import subprocess
import shutil
import os

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from typing import Optional

class AppInput(BaseAppInput):
    input_image: File = Field(description="The input image to animate")
    prompt: str = Field(description="Text prompt describing the desired animation or motion", default="make this image come alive, cinematic motion, smooth animation")
    negative_prompt: str = Field(
        description="Negative prompt to avoid unwanted elements",
        default="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards, watermark, text, signature"
    )
    duration_seconds: float = Field(description="Duration of the generated video in seconds", default=2.0, ge=0.3, le=3.4)
    guidance_scale: float = Field(description="Controls adherence to the prompt. Higher values = more adherence", default=1.0, ge=0.0, le=20.0)
    steps: int = Field(description="Number of inference steps. More steps = higher quality but slower", default=4, ge=1, le=30)
    seed: int = Field(description="Random seed for reproducible results", default=42, ge=0, le=2147483647)
    randomize_seed: bool = Field(description="Whether to use a random seed instead of the provided seed", default=False)
    bounce_loop: bool = Field(description="Create a bounce loop video (forward then backward)", default=False)

class AppOutput(BaseAppOutput):
    video_output: File = Field(description="Generated video file (.mp4)")

# Variant-specific configurations
configs = {
    "causvid_480p": {
        "model_id": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        "lora_repo_id": "Kijai/WanVideo_comfy",
        "lora_filename": "Wan21_CausVid_14B_T2V_lora_rank32.safetensors",
        "lora_adapter_name": "causvid_lora",
        "lora_weight": 0.95,
        "max_area": 480.0 * 832.0,  # 399,360 - original 480p area
        "default_height": 512,
        "default_width": 896,
        "max_height": 896,
        "max_width": 896
    },
    "causvid_720p": {
        "model_id": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",  # Use 480p model but with larger max area
        "lora_repo_id": "Kijai/WanVideo_comfy",
        "lora_filename": "Wan21_CausVid_14B_T2V_lora_rank32.safetensors",
        "lora_adapter_name": "causvid_lora",
        "lora_weight": 0.95,
        "max_area": 720.0 * 1280.0,  # 921,600 - 2.3x larger than 480p
        "default_height": 720,
        "default_width": 1280,
        "max_height": 1280,
        "max_width": 1280
    },
    "fusionx_480p": {
        "model_id": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        "lora_repo_id": "vrgamedevgirl84/Wan14BT2VFusioniX",
        "lora_filename": "FusionX_LoRa/Wan2.1_T2V_14B_FusionX_LoRA.safetensors",
        "lora_adapter_name": "fusionx_lora",
        "lora_weight": 0.95,
        "max_area": 480.0 * 832.0,  # 399,360 - original 480p area
        "default_height": 512,
        "default_width": 896,
        "max_height": 896,
        "max_width": 896
    },
    "fusionx_720p": {
        "model_id": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",  # Use 480p model but with larger max area
        "lora_repo_id": "vrgamedevgirl84/Wan14BT2VFusioniX",
        "lora_filename": "FusionX_LoRa/Wan2.1_T2V_14B_FusionX_LoRA.safetensors",
        "lora_adapter_name": "fusionx_lora",
        "lora_weight": 0.95,
        "max_area": 720.0 * 1280.0,  # 921,600 - 2.3x larger than 480p
        "default_height": 720,
        "default_width": 1280,
        "max_height": 1280,
        "max_width": 1280
    }
}

class App(BaseApp):
    def run_ffmpeg_command(self, command):
        """Runs an ffmpeg command using subprocess and raises an error if it fails."""
        print(f"Running ffmpeg command: {' '.join(command)}")
        process = subprocess.run(command, capture_output=True, text=True, check=True)
        return process

    def create_bounce_video(self, input_video_path):
        """Create a bounce loop video (forward then backward) using ffmpeg, skipping the first frame to avoid artifacts."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as bounce_file:
            bounce_output_path = bounce_file.name
        
        try:
            # Create the bounce effect: original (skip first frame) -> reversed
            # Skip first frame using trim filter to avoid first-frame artifacts
            bounce_command = [
                "ffmpeg",
                "-y",  # Overwrite output file
                "-i", input_video_path,
                "-filter_complex", 
                "[0:v]trim=start_frame=1,split[v1][v2];[v2]reverse[vr];[v1][vr]concat=n=2:v=1:a=0[vout]",
                "-map", "[vout]",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "18",
                "-preset", "medium",
                "-movflags", "+faststart",
                bounce_output_path
            ]
            self.run_ffmpeg_command(bounce_command)
            
            # Verify the output file was created and has content
            if not os.path.exists(bounce_output_path) or os.path.getsize(bounce_output_path) == 0:
                raise RuntimeError("Bounce video creation failed: output file is empty or doesn't exist")
            
            print(f"Bounce video created successfully: {os.path.getsize(bounce_output_path)} bytes")
            return bounce_output_path
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg bounce processing failed: {e.stderr}")
        except Exception as e:
            raise RuntimeError(f"An error occurred during bounce creation: {str(e)}")

    async def setup(self, metadata):
        """Initialize the Wan 2.1 model and resources with variant-specific configuration."""
        # Check if ffmpeg is available for bounce loop functionality
        if shutil.which("ffmpeg") is None:
            print("Warning: ffmpeg not found. Bounce loop functionality will be disabled.")
            self.ffmpeg_available = False
        else:
            self.ffmpeg_available = True
        
        # Get variant-specific configuration
        self.variant_config = configs[metadata.app_variant]
        print(f"Using variant: {metadata.app_variant}")
        print(f"Model: {self.variant_config['model_id']}")
        print(f"Max area: {self.variant_config['max_area']}")
        
        # Simple device detection without Accelerator to avoid meta tensor issues
        if torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        print(f"Using device: {self.device}")
        
        # Constants - with variant-specific values
        self.MOD_VALUE = 32
        self.DEFAULT_H_SLIDER_VALUE = self.variant_config["default_height"]
        self.DEFAULT_W_SLIDER_VALUE = self.variant_config["default_width"]
        self.NEW_FORMULA_MAX_AREA = self.variant_config["max_area"]
        self.SLIDER_MIN_H, self.SLIDER_MAX_H = 128, self.variant_config["max_height"]
        self.SLIDER_MIN_W, self.SLIDER_MAX_W = 128, self.variant_config["max_width"]
        self.MAX_SEED = np.iinfo(np.int32).max
        self.FIXED_FPS = 24
        self.MIN_FRAMES_MODEL = 8
        self.MAX_FRAMES_MODEL = 81

        # Load components with variant-specific model
        MODEL_ID = self.variant_config["model_id"]
        LORA_REPO_ID = self.variant_config["lora_repo_id"]
        LORA_FILENAME = self.variant_config["lora_filename"]
        LORA_ADAPTER_NAME = self.variant_config["lora_adapter_name"]
        LORA_WEIGHT = self.variant_config["lora_weight"]

        image_encoder = CLIPVisionModel.from_pretrained(MODEL_ID, subfolder="image_encoder", torch_dtype=torch.float32)
        vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
        pipe = WanImageToVideoPipeline.from_pretrained(
            MODEL_ID, vae=vae, image_encoder=image_encoder, torch_dtype=torch.bfloat16
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=8.0)
        
        # Move to device exactly as original
        pipe.to(self.device)

        # Load LoRA weights with variant-specific configuration
        lora_path = hf_hub_download(repo_id=LORA_REPO_ID, filename=LORA_FILENAME)
        pipe.load_lora_weights(lora_path, adapter_name=LORA_ADAPTER_NAME)
        pipe.set_adapters([LORA_ADAPTER_NAME], adapter_weights=[LORA_WEIGHT])
        pipe.fuse_lora()

        self.pipe = pipe

    def _calculate_new_dimensions_wan(self, pil_image, mod_val, calculation_max_area, min_slider_h, max_slider_h, min_slider_w, max_slider_w, default_h, default_w):
        """Calculate new dimensions ensuring they don't exceed the original image dimensions."""
        orig_w, orig_h = pil_image.size
        if orig_w <= 0 or orig_h <= 0:
            return default_h, default_w
            
        aspect_ratio = orig_h / orig_w
        calc_h = round(np.sqrt(calculation_max_area * aspect_ratio))
        calc_w = round(np.sqrt(calculation_max_area / aspect_ratio))
        
        calc_h = max(mod_val, (calc_h // mod_val) * mod_val)
        calc_w = max(mod_val, (calc_w // mod_val) * mod_val)
        
        # Ensure dimensions don't exceed original image dimensions (rounded down to mod_val multiple)
        max_allowed_h = (orig_h // mod_val) * mod_val
        max_allowed_w = (orig_w // mod_val) * mod_val
        
        # Clip to both slider constraints and original image constraints
        effective_max_h = min(max_slider_h // mod_val * mod_val, max_allowed_h)
        effective_max_w = min(max_slider_w // mod_val * mod_val, max_allowed_w)
        
        new_h = int(np.clip(calc_h, min_slider_h, effective_max_h))
        new_w = int(np.clip(calc_w, min_slider_w, effective_max_w))
        
        return new_h, new_w

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Generate video from input image using Wan 2.1 I2V model exactly as in original."""
        
        # Check if input file exists
        if not input_data.input_image.exists():
            raise RuntimeError(f"Input image does not exist at path: {input_data.input_image.path}")
        
        # Load and process input image
        input_image = Image.open(input_data.input_image.path).convert("RGB")
        
        # Calculate optimal dimensions automatically based on input image aspect ratio and variant constraints
        target_h, target_w = self._calculate_new_dimensions_wan(
            input_image,
            self.MOD_VALUE,
            self.NEW_FORMULA_MAX_AREA,
            self.SLIDER_MIN_H,
            self.SLIDER_MAX_H,
            self.SLIDER_MIN_W,
            self.SLIDER_MAX_W,
            self.DEFAULT_H_SLIDER_VALUE,
            self.DEFAULT_W_SLIDER_VALUE
        )
        
        print(f"Computed optimal dimensions: {target_w}x{target_h} (from input {input_image.size[0]}x{input_image.size[1]})")
        
        # Calculate number of frames exactly as original
        num_frames = np.clip(
            int(round(input_data.duration_seconds * self.FIXED_FPS)),
            self.MIN_FRAMES_MODEL,
            self.MAX_FRAMES_MODEL
        )
        
        # Set seed exactly as original
        current_seed = random.randint(0, self.MAX_SEED) if input_data.randomize_seed else int(input_data.seed)
        
        # Resize image to target dimensions
        resized_image = input_image.resize((target_w, target_h))
        
        # Generate video exactly as original
        with torch.inference_mode():
            output_frames_list = self.pipe(
                image=resized_image,
                prompt=input_data.prompt,
                negative_prompt=input_data.negative_prompt,
                height=target_h,
                width=target_w,
                num_frames=num_frames,
                guidance_scale=float(input_data.guidance_scale),
                num_inference_steps=int(input_data.steps),
                generator=torch.Generator(device=self.device).manual_seed(current_seed)
            ).frames[0]
        
        # Export video to temporary file exactly as original
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:
            video_path = tmpfile.name
            export_to_video(output_frames_list, video_path, fps=self.FIXED_FPS)
        
        # Create bounce video if requested and ffmpeg is available
        final_video_path = video_path
        if input_data.bounce_loop:
            if self.ffmpeg_available:
                print("Creating bounce loop video...")
                bounce_video_path = self.create_bounce_video(video_path)
                final_video_path = bounce_video_path
            else:
                print("Warning: Bounce loop requested but ffmpeg is not available. Returning original video.")
        
        return AppOutput(
            video_output=File(path=final_video_path),
        )

    async def unload(self):
        """Clean up GPU resources."""
        if hasattr(self, 'pipe'):
            del self.pipe
        torch.cuda.empty_cache()
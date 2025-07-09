import torch
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, UniPCMultistepScheduler
from diffusers.utils import export_to_video
from transformers import CLIPVisionModel
import tempfile
from huggingface_hub import hf_hub_download
import numpy as np
from PIL import Image
import random
from accelerate import Accelerator

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from typing import Optional


class AppInput(BaseAppInput):
    input_image: File = Field(description="The input image to animate")
    prompt: str = Field(
        description="Text prompt describing the desired animation or motion",
        default="make this image come alive, cinematic motion, smooth animation",
    )
    height: int = Field(
        description="Target height for the output video (will be adjusted to multiple of 32)",
        default=512,
        ge=128,
        le=896,
    )
    width: int = Field(
        description="Target width for the output video (will be adjusted to multiple of 32)",
        default=896,
        ge=128,
        le=896,
    )
    negative_prompt: str = Field(
        description="Negative prompt to avoid unwanted elements",
        default="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards, watermark, text, signature",
    )
    duration_seconds: float = Field(
        description="Duration of the generated video in seconds",
        default=2.0,
        ge=0.3,
        le=3.4,
    )
    guidance_scale: float = Field(
        description="Controls adherence to the prompt. Higher values = more adherence",
        default=1.0,
        ge=0.0,
        le=20.0,
    )
    steps: int = Field(
        description="Number of inference steps. More steps = higher quality but slower",
        default=4,
        ge=1,
        le=30,
    )
    seed: int = Field(
        description="Random seed for reproducible results",
        default=42,
        ge=0,
        le=2147483647,
    )
    randomize_seed: bool = Field(
        description="Whether to use a random seed instead of the provided seed",
        default=False,
    )


class AppOutput(BaseAppOutput):
    video_output: File = Field(description="Generated video file (.mp4)")
    seed_used: int = Field(description="The seed used for generation")


class App(BaseApp):
    async def setup(self, metadata):
        """Initialize the Wan 2.1 model and resources exactly as in original."""
        # Simple device detection without Accelerator to avoid meta tensor issues
        self.device = Accelerator().device
        print(f"Using device: {self.device}")

        # Model configuration - exactly as in original
        MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
        LORA_REPO_ID = "Kijai/WanVideo_comfy"
        LORA_FILENAME = "Wan21_CausVid_14B_T2V_lora_rank32.safetensors"

        # Constants - exactly as in original
        self.MOD_VALUE = 32
        self.DEFAULT_H_SLIDER_VALUE = 512
        self.DEFAULT_W_SLIDER_VALUE = 896
        self.NEW_FORMULA_MAX_AREA = 480.0 * 832.0
        self.SLIDER_MIN_H, self.SLIDER_MAX_H = 128, 896
        self.SLIDER_MIN_W, self.SLIDER_MAX_W = 128, 896
        self.MAX_SEED = np.iinfo(np.int32).max
        self.FIXED_FPS = 24
        self.MIN_FRAMES_MODEL = 8
        self.MAX_FRAMES_MODEL = 81

        # Load components exactly as in original
        image_encoder = CLIPVisionModel.from_pretrained(
            MODEL_ID, subfolder="image_encoder", torch_dtype=torch.float32
        )
        vae = AutoencoderKLWan.from_pretrained(
            MODEL_ID, subfolder="vae", torch_dtype=torch.float32
        )
        pipe = WanImageToVideoPipeline.from_pretrained(
            MODEL_ID, vae=vae, image_encoder=image_encoder, torch_dtype=torch.bfloat16
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config, flow_shift=8.0
        )

        pipe.to(self.device)

        # Load LoRA weights exactly as original
        causvid_path = hf_hub_download(repo_id=LORA_REPO_ID, filename=LORA_FILENAME)
        pipe.load_lora_weights(causvid_path, adapter_name="causvid_lora")
        pipe.set_adapters(["causvid_lora"], adapter_weights=[0.95])
        pipe.fuse_lora()

        
        self.pipe = pipe

    def _calculate_new_dimensions_wan(
        self,
        pil_image,
        mod_val,
        calculation_max_area,
        min_slider_h,
        max_slider_h,
        min_slider_w,
        max_slider_w,
        default_h,
        default_w,
    ):
        """Calculate new dimensions exactly as in original code."""
        orig_w, orig_h = pil_image.size
        if orig_w <= 0 or orig_h <= 0:
            return default_h, default_w

        aspect_ratio = orig_h / orig_w
        calc_h = round(np.sqrt(calculation_max_area * aspect_ratio))
        calc_w = round(np.sqrt(calculation_max_area / aspect_ratio))

        calc_h = max(mod_val, (calc_h // mod_val) * mod_val)
        calc_w = max(mod_val, (calc_w // mod_val) * mod_val)

        new_h = int(np.clip(calc_h, min_slider_h, (max_slider_h // mod_val) * mod_val))
        new_w = int(np.clip(calc_w, min_slider_w, (max_slider_w // mod_val) * mod_val))

        return new_h, new_w

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Generate video from input image using Wan 2.1 I2V model exactly as in original."""

        # Check if input file exists
        if not input_data.input_image.exists():
            raise RuntimeError(
                f"Input image does not exist at path: {input_data.input_image.path}"
            )

        # Load and process input image
        input_image = Image.open(input_data.input_image.path).convert("RGB")

        # Calculate target dimensions using original formula
        target_h, target_w = self._calculate_new_dimensions_wan(
            input_image,
            self.MOD_VALUE,
            self.NEW_FORMULA_MAX_AREA,
            self.SLIDER_MIN_H,
            self.SLIDER_MAX_H,
            self.SLIDER_MIN_W,
            self.SLIDER_MAX_W,
            input_data.height,
            input_data.width,
        )

        # Override with user-specified dimensions if they want exact control
        # But ensure they're multiples of MOD_VALUE
        target_h = max(
            self.MOD_VALUE, (int(input_data.height) // self.MOD_VALUE) * self.MOD_VALUE
        )
        target_w = max(
            self.MOD_VALUE, (int(input_data.width) // self.MOD_VALUE) * self.MOD_VALUE
        )

        # Calculate number of frames exactly as original
        num_frames = np.clip(
            int(round(input_data.duration_seconds * self.FIXED_FPS)),
            self.MIN_FRAMES_MODEL,
            self.MAX_FRAMES_MODEL,
        )

        # Set seed exactly as original
        current_seed = (
            random.randint(0, self.MAX_SEED)
            if input_data.randomize_seed
            else int(input_data.seed)
        )

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
                generator=torch.Generator(device=self.device).manual_seed(current_seed),
            ).frames[0]

        # Export video to temporary file exactly as original
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:
            video_path = tmpfile.name
            export_to_video(output_frames_list, video_path, fps=self.FIXED_FPS)

        return AppOutput(video_output=File(path=video_path), seed_used=current_seed)

    async def unload(self):
        """Clean up GPU resources."""
        if hasattr(self, "pipe"):
            del self.pipe
        torch.cuda.empty_cache()

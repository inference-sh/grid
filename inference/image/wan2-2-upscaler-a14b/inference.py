import os
import logging
import gc
import tempfile
from typing import Optional

import numpy as np
import torch
from pydantic import Field
from PIL import Image
from huggingface_hub import hf_hub_download
from accelerate import Accelerator
from diffusers import WanImageToVideoPipeline, WanTransformer3DModel, GGUFQuantizationConfig
from diffusers.hooks import apply_group_offloading

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File

# Enable HF Hub fast transfer for faster model downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


# Model variants mapping for GGUF quantization from QuantStack
# Only includes variants where both HighNoise and LowNoise transformers are available
MODEL_VARIANTS = {
    "default": None,  # Use default F16 model
    "q2_k": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q2_K.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q2_K.gguf",
    },
    "q3_k_s": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q3_K_S.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q3_K_S.gguf",
    },
    "q3_k_m": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q3_K_M.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q3_K_M.gguf",
    },
    "q4_k_s": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q4_K_S.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q4_K_S.gguf",
    },
    "q4_k_m": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q4_K_M.gguf",
    },
    "q5_0": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q5_0.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q5_0.gguf",
    },
    "q5_1": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q5_1.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q5_1.gguf",
    },
    "q5_k_s": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q5_K_S.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q5_K_S.gguf",
    },
    "q5_k_m": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q5_K_M.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q5_K_M.gguf",
    },
    "q6_k": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q6_K.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q6_K.gguf",
    },
    "q8_0": {
        "high_noise": "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q8_0.gguf",
        "low_noise": "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q8_0.gguf",
    },
}

DEFAULT_VARIANT = "default"


class AppInput(BaseAppInput):
    image: File = Field(description="Input image to upscale using the Wan2.2 I2V pipeline")
    scale: float = Field(description="Scale factor (e.g., 2.0 doubles width and height)")
    seed: Optional[int] = Field(default=None, description="Optional random seed for reproducibility")


class AppOutput(BaseAppOutput):
    file: File = Field(description="Generated image file")


class App(BaseApp):
    async def setup(self, metadata):
        """Initialize the Wan2.2 Image-to-Video pipeline and resources here."""
        # Initialize accelerator
        self.accelerator = Accelerator()

        # Set up device and dtype using accelerator
        self.device = self.accelerator.device
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        # Get variant and determine if using quantization/offloading
        variant = getattr(metadata, "app_variant", DEFAULT_VARIANT)
        # Offloading policy: default always uses CPU offload; explicit lowvram uses group offload
        use_group_offload = variant.endswith("_offload_lowvram")
        # Strip suffix for base variant lookup
        base_variant = variant.replace("_offload_lowvram", "").replace("_offload", "")
        if base_variant not in MODEL_VARIANTS:
            logging.warning(
                f"Unknown variant '{variant}', falling back to default '{DEFAULT_VARIANT}'"
            )
            base_variant = DEFAULT_VARIANT

        # Load pipeline based on variant
        if base_variant == "default":
            # Load standard F16 pipeline
            self.pipe = WanImageToVideoPipeline.from_pretrained(
                "Wan-AI/Wan2.2-I2V-A14B-Diffusers", torch_dtype=self.dtype
            )
            # Apply offloading/device placement
            if use_group_offload:
                onload_device = self.device
                offload_device = torch.device("cpu")
                # Group offloading on key modules
                self.pipe.vae.enable_group_offload(
                    onload_device=onload_device,
                    offload_device=offload_device,
                    offload_type="leaf_level",
                )
                self.pipe.transformer.enable_group_offload(
                    onload_device=onload_device,
                    offload_device=offload_device,
                    offload_type="leaf_level",
                )
                if hasattr(self.pipe, "transformer_2"):
                    self.pipe.transformer_2.enable_group_offload(
                        onload_device=onload_device,
                        offload_device=offload_device,
                        offload_type="leaf_level",
                    )
                apply_group_offloading(
                    self.pipe.text_encoder,
                    onload_device=onload_device,
                    offload_device=offload_device,
                    offload_type="leaf_level",
                )
            else:
                # Default: enable CPU offload for memory efficiency
                self.pipe.enable_model_cpu_offload()
        else:
            # Load quantized transformers
            repo_id = "QuantStack/Wan2.2-I2V-A14B-GGUF"
            variant_files = MODEL_VARIANTS[base_variant]

            # Download and load high noise transformer (main transformer)
            high_noise_path = hf_hub_download(
                repo_id=repo_id, filename=variant_files["high_noise"]
            )

            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            transformer_high_noise = WanTransformer3DModel.from_single_file(
                high_noise_path,
                quantization_config=GGUFQuantizationConfig(compute_dtype=self.dtype),
                config="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
                subfolder="transformer",
                torch_dtype=self.dtype,
            )

            # Download and load low noise transformer (transformer_2)
            low_noise_path = hf_hub_download(
                repo_id=repo_id, filename=variant_files["low_noise"]
            )

            # Force garbage collection again
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            transformer_low_noise = WanTransformer3DModel.from_single_file(
                low_noise_path,
                quantization_config=GGUFQuantizationConfig(compute_dtype=self.dtype),
                config="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
                subfolder="transformer_2",
                torch_dtype=self.dtype,
            )

            # Load pipeline with quantized transformers
            self.pipe = WanImageToVideoPipeline.from_pretrained(
                "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
                transformer=transformer_high_noise,  # High noise goes to main transformer
                transformer_2=transformer_low_noise,  # Low noise goes to transformer_2
                torch_dtype=self.dtype,
            )
            # Apply offloading/device placement
            if use_group_offload:
                onload_device = self.device
                offload_device = torch.device("cpu")
                self.pipe.vae.enable_group_offload(
                    onload_device=onload_device,
                    offload_device=offload_device,
                    offload_type="leaf_level",
                )
                self.pipe.transformer.enable_group_offload(
                    onload_device=onload_device,
                    offload_device=offload_device,
                    offload_type="leaf_level",
                )
                if hasattr(self.pipe, "transformer_2"):
                    self.pipe.transformer_2.enable_group_offload(
                        onload_device=onload_device,
                        offload_device=offload_device,
                        offload_type="leaf_level",
                    )
                apply_group_offloading(
                    self.pipe.text_encoder,
                    onload_device=onload_device,
                    offload_device=offload_device,
                    offload_type="leaf_level",
                )
            else:
                # Default: enable CPU offload for memory efficiency
                self.pipe.enable_model_cpu_offload()

    def _round_to_valid_dims(self, width: int, height: int) -> tuple[int, int]:
        """Round width/height down to the nearest valid dimensions required by the pipeline."""
        mod_value = (
            self.pipe.vae_scale_factor_spatial * self.pipe.transformer.config.patch_size[1]
        )
        width = max(mod_value, (width // mod_value) * mod_value)
        height = max(mod_value, (height // mod_value) * mod_value)
        return int(width), int(height)

    def _to_pil(self, frame) -> Image.Image:
        """Convert a pipeline frame (numpy array or PIL) to a PIL Image."""
        if isinstance(frame, Image.Image):
            return frame
        # Assume numpy array
        arr = frame
        if arr.dtype != np.uint8:
            max_val = float(arr.max()) if hasattr(arr, "max") else 1.0
            if max_val <= 1.0:
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        # Load input image
        with Image.open(input_data.image.path) as pil_image:
            pil_image = pil_image.convert("RGB")

        # Compute scaled target size and round to valid dims
        scale_factor = float(input_data.scale)
        if not (scale_factor > 0):
            scale_factor = 1.0
        desired_width = int(round(pil_image.width * scale_factor))
        desired_height = int(round(pil_image.height * scale_factor))
        width, height = self._round_to_valid_dims(desired_width, desired_height)

        # Optional seed
        generator = None
        if input_data.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(input_data.seed)

        # Generate a single frame using the I2V pipeline
        try:
            with torch.inference_mode():
                frames = self.pipe(
                    image=pil_image,
                    prompt="",
                    negative_prompt="",
                    height=height,
                    width=width,
                    num_frames=1,
                    guidance_scale=3.5,
                    num_inference_steps=40,
                    generator=generator,
                    last_image=None,
                ).frames[0]

            first_frame = self._to_pil(frames[0])

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                output_path = temp_file.name

            first_frame.save(output_path, format="PNG")

            return AppOutput(file=File(path=output_path))
        finally:
            try:
                del pil_image
                del generator
                del frames
                del first_frame
            except Exception:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass

    async def unload(self):
        """Clean up resources here."""
        if hasattr(self, "pipe"):
            del self.pipe
        if hasattr(self, "device") and self.device.type == "cuda":
            torch.cuda.empty_cache()
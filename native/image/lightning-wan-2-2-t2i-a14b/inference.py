import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import torch
import numpy as np
import tempfile
from typing import Optional
from pydantic import Field
from PIL import Image
from diffusers import AutoencoderKLWan, WanTransformer3DModel, GGUFQuantizationConfig, WanPipeline, UniPCMultistepScheduler
from huggingface_hub import hf_hub_download
from accelerate import Accelerator
from diffusers.hooks import apply_group_offloading

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File

MODEL_VARIANTS = {
    "default": None,
    "q2_k": {
        "high_noise": "HighNoise/Wan2.2-T2V-A14B-HighNoise-Q2_K.gguf",
        "low_noise": "LowNoise/Wan2.2-T2V-A14B-LowNoise-Q2_K.gguf"
    },
    "q3_k_s": {
        "high_noise": "HighNoise/Wan2.2-T2V-A14B-HighNoise-Q3_K_S.gguf",
        "low_noise": "LowNoise/Wan2.2-T2V-A14B-LowNoise-Q3_K_S.gguf"
    },
    "q3_k_m": {
        "high_noise": "HighNoise/Wan2.2-T2V-A14B-HighNoise-Q3_K_M.gguf",
        "low_noise": "LowNoise/Wan2.2-T2V-A14B-LowNoise-Q3_K_M.gguf"
    },
    "q4_0": {
        "high_noise": "HighNoise/Wan2.2-T2V-A14B-HighNoise-Q4_0.gguf",
        "low_noise": "LowNoise/Wan2.2-T2V-A14B-LowNoise-Q4_0.gguf"
    },
    "q4_1": {
        "high_noise": "HighNoise/Wan2.2-T2V-A14B-HighNoise-Q4_1.gguf",
        "low_noise": "LowNoise/Wan2.2-T2V-A14B-LowNoise-Q4_1.gguf"
    },
    "q4_k_m": {
        "high_noise": "HighNoise/Wan2.2-T2V-A14B-HighNoise-Q4_K_M.gguf",
        "low_noise": "LowNoise/Wan2.2-T2V-A14B-LowNoise-Q4_K_M.gguf"
    },
    "q4_k_s": {
        "high_noise": "HighNoise/Wan2.2-T2V-A14B-HighNoise-Q4_K_S.gguf",
        "low_noise": "LowNoise/Wan2.2-T2V-A14B-LowNoise-Q4_K_S.gguf"
    },
    "q5_0": {
        "high_noise": "HighNoise/Wan2.2-T2V-A14B-HighNoise-Q5_0.gguf",
        "low_noise": "LowNoise/Wan2.2-T2V-A14B-LowNoise-Q5_0.gguf"
    },
    "q5_1": {
        "high_noise": "HighNoise/Wan2.2-T2V-A14B-HighNoise-Q5_1.gguf",
        "low_noise": "LowNoise/Wan2.2-T2V-A14B-LowNoise-Q5_1.gguf"
    },
    "q5_k_m": {
        "high_noise": "HighNoise/Wan2.2-T2V-A14B-HighNoise-Q5_K_M.gguf",
        "low_noise": "LowNoise/Wan2.2-T2V-A14B-LowNoise-Q5_K_M.gguf"
    },
    "q5_k_s": {
        "high_noise": "HighNoise/Wan2.2-T2V-A14B-HighNoise-Q5_K_S.gguf",
        "low_noise": "LowNoise/Wan2.2-T2V-A14B-LowNoise-Q5_K_S.gguf"
    },
    "q6_k": {
        "high_noise": "HighNoise/Wan2.2-T2V-A14B-HighNoise-Q6_K.gguf",
        "low_noise": "LowNoise/Wan2.2-T2V-A14B-LowNoise-Q6_K.gguf"
    },
    "q8_0": {
        "high_noise": "HighNoise/Wan2.2-T2V-A14B-HighNoise-Q8_0.gguf",
        "low_noise": "LowNoise/Wan2.2-T2V-A14B-LowNoise-Q8_0.gguf"
    }
}

DEFAULT_VARIANT = "default"

class AppInput(BaseAppInput):
    prompt: str = Field(description="Text prompt for image generation")
    negative_prompt: str = Field(default="oversaturated, overexposed, static, blurry details, subtitles, stylized, artwork, painting, still image, overall gray, worst quality, low quality, JPEG artifacts, ugly, deformed, extra fingers, poorly drawn hands, poorly drawn face, malformed, disfigured, deformed limbs, fused fingers, static motionless frame, cluttered background, three legs, crowded background, walking backwards")
    width: Optional[int] = Field(default=1024)
    height: Optional[int] = Field(default=1024)
    num_inference_steps: int = Field(default=4)
    seed: Optional[int] = Field(default=None)
    boundary_ratio: float = Field(default=0.875, ge=0.0, le=1.0)

class AppOutput(BaseAppOutput):
    image: File = Field(description="output image")

class App(BaseApp):
    async def setup(self, metadata):
        print("Setting up Wan2.2 T2I with Lightning LoRA...")
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        variant = getattr(metadata, "app_variant", DEFAULT_VARIANT)
        if variant not in MODEL_VARIANTS and not any(variant.endswith(suf) for suf in ["_offload", "_offload_lowvram"]):
            variant = DEFAULT_VARIANT

        use_cpu_offload = variant.endswith("_offload")
        use_group_offload = variant.endswith("_offload_lowvram")
        base_variant = variant.replace("_offload_lowvram", "").replace("_offload", "")

        self.vae = AutoencoderKLWan.from_pretrained(
            "Wan-AI/Wan2.1-T2V-14B-Diffusers", subfolder="vae", torch_dtype=torch.float32
        )

        if base_variant == "default":
            transformer_high_noise = WanTransformer3DModel.from_pretrained(
                "Wan-AI/Wan2.2-T2V-A14B-Diffusers", subfolder="transformer", torch_dtype=self.dtype
            )
            transformer_low_noise = WanTransformer3DModel.from_pretrained(
                "Wan-AI/Wan2.2-T2V-A14B-Diffusers", subfolder="transformer_2", torch_dtype=self.dtype
            )
            self.pipe = WanPipeline.from_pretrained(
                "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                vae=self.vae,
                transformer=transformer_high_noise,
                transformer_2=transformer_low_noise,
                boundary_ratio=0.875,
                torch_dtype=self.dtype,
            )
            # Configure UniPCM scheduler with flow_shift=8.0 for faster generation
            self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config, flow_shift=8.0)
            if use_cpu_offload:
                self.pipe.enable_model_cpu_offload()
            elif use_group_offload:
                onload_device = self.device
                offload_device = torch.device("cpu")
                self.pipe.vae.enable_group_offload(onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
                self.pipe.transformer.enable_group_offload(onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
                if hasattr(self.pipe, 'transformer_2'):
                    self.pipe.transformer_2.enable_group_offload(onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
                apply_group_offloading(self.pipe.text_encoder, onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
            else:
                # No offloading: move entire pipeline to accelerator device
                self.pipe.to(self.device)
        else:
            repo_id = "QuantStack/Wan2.2-T2V-A14B-GGUF"
            variant_files = MODEL_VARIANTS[base_variant]
            high_noise_path = hf_hub_download(repo_id=repo_id, filename=variant_files["high_noise"])
            low_noise_path = hf_hub_download(repo_id=repo_id, filename=variant_files["low_noise"])
            transformer_high_noise = WanTransformer3DModel.from_single_file(
                high_noise_path,
                quantization_config=GGUFQuantizationConfig(compute_dtype=self.dtype),
                config="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                subfolder="transformer",
                torch_dtype=self.dtype,
            )
            transformer_low_noise = WanTransformer3DModel.from_single_file(
                low_noise_path,
                quantization_config=GGUFQuantizationConfig(compute_dtype=self.dtype),
                config="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                subfolder="transformer_2",
                torch_dtype=self.dtype,
            )
            self.pipe = WanPipeline.from_pretrained(
                "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                vae=self.vae,
                transformer=transformer_high_noise,
                transformer_2=transformer_low_noise,
                boundary_ratio=0.875,
                torch_dtype=self.dtype,
            )
            # Configure UniPCM scheduler with flow_shift=8.0 for faster generation
            self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config, flow_shift=8.0)
            if use_cpu_offload:
                self.pipe.enable_model_cpu_offload()
            elif use_group_offload:
                onload_device = self.device
                offload_device = torch.device("cpu")
                self.pipe.vae.enable_group_offload(onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
                self.pipe.transformer.enable_group_offload(onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
                if hasattr(self.pipe, 'transformer_2'):
                    self.pipe.transformer_2.enable_group_offload(onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
                apply_group_offloading(self.pipe.text_encoder, onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
            else:
                # No offloading: move entire pipeline to accelerator device
                self.pipe.to(self.device)

        # Load Lightning LoRAs on both transformers
        self.pipe.load_lora_weights(
            "Kijai/WanVideo_comfy",
            weight_name="Wan22-Lightning/Wan2.2-Lightning_T2V-A14B-4steps-lora_HIGH_fp16.safetensors",
            adapter_name="lightning",
            load_into_transformer=True,
            load_into_transformer_2=False,
        )
        self.pipe.load_lora_weights(
            "Kijai/WanVideo_comfy",
            weight_name="Wan22-Lightning/Wan2.2-Lightning_T2V-A14B-4steps-lora_LOW_fp16.safetensors",
            adapter_name="lightning_2",
            load_into_transformer=False,
            load_into_transformer_2=True,
        )
        self.pipe.set_adapters(["lightning", "lightning_2"], adapter_weights=[1.25, 1.25])

        print("Setup complete!")

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        if input_data.seed is not None:
            torch.manual_seed(input_data.seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed(input_data.seed)

        try:
            self.pipe.register_to_config(boundary_ratio=input_data.boundary_ratio)
        except Exception:
            pass

        with torch.inference_mode():
            output = self.pipe(
                prompt=input_data.prompt,
                negative_prompt=input_data.negative_prompt,
                height=input_data.height,
                width=input_data.width,
                num_frames=1,
                guidance_scale=1.0,
                num_inference_steps=input_data.num_inference_steps,
            ).frames[0]

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            output_path = temp_file.name

        frame = output[0] if isinstance(output, list) and len(output) > 0 else output
        if hasattr(frame, 'save'):
            frame.save(output_path)
            return AppOutput(image=File(path=output_path))

        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()
        frame = np.squeeze(frame)
        if len(frame.shape) == 3 and frame.shape[0] == 3:
            frame = frame.transpose(1, 2, 0)
        frame = np.clip(frame, 0, 1)
        frame = (frame * 255).astype(np.uint8)
        Image.fromarray(frame).save(output_path)
        return AppOutput(image=File(path=output_path))


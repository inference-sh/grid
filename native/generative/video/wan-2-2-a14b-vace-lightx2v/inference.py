import os
import torch
import tempfile
import PIL.Image
import cv2
from diffusers import AutoencoderKLWan, WanVACEPipeline, GGUFQuantizationConfig, WanTransformer3DModel
from diffusers.utils import export_to_video, load_image
from huggingface_hub import hf_hub_download
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from typing import List, Optional
from accelerate import Accelerator

# Enable faster downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Model variants mapping for GGUF quantization from QuantStack for VACE
MODEL_VARIANTS = {
    "default": None,  # Use default F16 model
    "low_vram": None,  # Use default F16 model with offloading
    "q4_0": {
        "high_noise": "HighNoise/Wan2.2-VACE-Fun-A14B-high-noise-Q4_0.gguf",
        "low_noise": "LowNoise/Wan2.2-VACE-Fun-A14B-low-noise-Q4_0.gguf"
    },
    "q4_0_offload": {
        "high_noise": "HighNoise/Wan2.2-VACE-Fun-A14B-high-noise-Q4_0.gguf",
        "low_noise": "LowNoise/Wan2.2-VACE-Fun-A14B-low-noise-Q4_0.gguf"
    },
    "q8_0": {
        "high_noise": "HighNoise/Wan2.2-VACE-Fun-A14B-high-noise-Q8_0.gguf",
        "low_noise": "LowNoise/Wan2.2-VACE-Fun-A14B-low-noise-Q8_0.gguf"
    },
    "q8_0_offload": {
        "high_noise": "HighNoise/Wan2.2-VACE-Fun-A14B-high-noise-Q8_0.gguf",
        "low_noise": "LowNoise/Wan2.2-VACE-Fun-A14B-low-noise-Q8_0.gguf"
    },
    "q4_0_cpu": {
        "high_noise": "HighNoise/Wan2.2-VACE-Fun-A14B-high-noise-Q4_0.gguf",
        "low_noise": "LowNoise/Wan2.2-VACE-Fun-A14B-low-noise-Q4_0.gguf"
    },
    "q8_0_cpu": {
        "high_noise": "HighNoise/Wan2.2-VACE-Fun-A14B-high-noise-Q8_0.gguf",
        "low_noise": "LowNoise/Wan2.2-VACE-Fun-A14B-low-noise-Q8_0.gguf"
    }
}

DEFAULT_VARIANT = "default"

def prepare_video_and_mask(height: int, width: int, num_frames: int, img: PIL.Image.Image = None):
    """Prepare video frames and mask for WAN VACE pipeline (legacy function for image-to-video)."""
    if img is not None:
        img = img.resize((width, height))
        frames = [img]
        # Ideally, this should be 127.5 to match original code, but they perform computation on numpy arrays
        # whereas we are passing PIL images. If you choose to pass numpy arrays, you can set it to 127.5 to
        # match the original code.
        frames.extend([PIL.Image.new("RGB", (width, height), (128, 128, 128))] * (num_frames - 1))
        mask_black = PIL.Image.new("L", (width, height), 0)
        mask_white = PIL.Image.new("L", (width, height), 255)
        mask = [mask_black, *[mask_white] * (num_frames - 1)]
    else:
        frames = []
        # Ideally, this should be 127.5 to match original code, but they perform computation on numpy arrays
        # whereas we are passing PIL images. If you choose to pass numpy arrays, you can set it to 127.5 to
        # match the original code.
        frames.extend([PIL.Image.new("RGB", (width, height), (128, 128, 128))] * (num_frames))
        mask_white = PIL.Image.new("L", (width, height), 255)
        mask = [mask_white] * (num_frames)
    return frames, mask

class AppInput(BaseAppInput):
    video: File = Field(description="Input video file to be processed and transformed")
    prompt: str = Field(description="Text prompt describing the desired video transformation")
    reference_images: Optional[List[File]] = Field(None, description="Optional list of reference images")
    negative_prompt: Optional[str] = Field(None, description="Negative prompt to avoid unwanted content")
    num_inference_steps: int = Field(default=8, ge=6, le=10, description="Number of inference steps (6-10 recommended)")
    boundary_ratio: float = Field(default=0.875, ge=0.0, le=1.0, description="Boundary ratio between high and low noise transformers (0-1)")
    seed: int = Field(default=42, description="Random seed for reproducible results")

class AppOutput(BaseAppOutput):
    video: File = Field(description="Generated video file")


class App(BaseApp):
    async def setup(self, metadata):
        """Initialize WAN 2.2 VACE pipeline with LightX2V LoRA."""
        # Setup device management
        self.accelerator = Accelerator()
        self.device = self.accelerator.device

        # Get variant from metadata
        variant = getattr(metadata, "app_variant", DEFAULT_VARIANT)
        if variant not in MODEL_VARIANTS:
            print(f"Unknown variant '{variant}', falling back to default '{DEFAULT_VARIANT}'")
            variant = DEFAULT_VARIANT

        print(f"Loading model variant: {variant}")

        # Set model repository
        self.model_id = "linoyts/Wan2.2-VACE-Fun-14B-diffusers"

        # Load VAE separately with float32 for stability
        print("Loading VAE...")
        self.vae = AutoencoderKLWan.from_pretrained(
            self.model_id,
            subfolder="vae",
            torch_dtype=torch.float32
        )

        # Determine offloading strategy from variant suffix
        use_cpu_offload = (variant.endswith("_offload") or variant.endswith("_cpu") or variant == "low_vram")
        is_cpu = variant.endswith("_cpu")

        if variant in ["default", "low_vram"]:
            # Load standard F16 pipeline
            print(f"Loading standard F16 WAN VACE pipeline for {variant}...")
            self.pipe = WanVACEPipeline.from_pretrained(
                self.model_id,
                vae=self.vae,
                torch_dtype=torch.bfloat16,
            )
        else:
            # Load quantized transformers
            print(f"Loading quantized transformers for {variant}...")
            repo_id = "QuantStack/Wan2.2-VACE-Fun-A14B-GGUF"
            variant_files = MODEL_VARIANTS[variant]

            # Download and load high noise transformer
            high_noise_path = hf_hub_download(repo_id=repo_id, filename=variant_files['high_noise'])
            transformer_high_noise = WanTransformer3DModel.from_single_file(
                high_noise_path,
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
                config=self.model_id,
                subfolder="transformer",
                torch_dtype=torch.bfloat16,
            )

            # Download and load low noise transformer
            low_noise_path = hf_hub_download(repo_id=repo_id, filename=variant_files['low_noise'])
            transformer_low_noise = WanTransformer3DModel.from_single_file(
                low_noise_path,
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
                config=self.model_id,
                subfolder="transformer_2",
                torch_dtype=torch.bfloat16,
            )

            # Create pipeline with both transformers
            self.pipe = WanVACEPipeline.from_pretrained(
                self.model_id,
                vae=self.vae,
                transformer=transformer_high_noise,  # High noise goes to main transformer
                transformer_2=transformer_low_noise,  # Low noise goes to transformer_2
                torch_dtype=torch.bfloat16,
            )

        # Handle offloading
        if use_cpu_offload:
            print("Enabling CPU offload...")
            self.pipe.enable_model_cpu_offload()
        elif not is_cpu:
            print(f"Moving pipeline to {self.device}...")
            self.pipe = self.pipe.to(self.device)

        # Load LightX2V LoRA weights - first instance for transformer
        print("Loading LightX2V LoRA weights for transformer...")
        self.pipe.load_lora_weights(
            "Kijai/WanVideo_comfy",
            weight_name="Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors",
            adapter_name="lightx2v"
        )

        # Load LightX2V LoRA weights - second instance for transformer_2
        print("Loading LightX2V LoRA weights for transformer_2...")
        kwargs_lora = {}
        kwargs_lora["load_into_transformer_2"] = True
        self.pipe.load_lora_weights(
            "Kijai/WanVideo_comfy",
            weight_name="Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors",
            adapter_name="lightx2v_2",
            **kwargs_lora
        )

        # Set adapters with weights
        print("Setting LoRA adapters...")
        self.pipe.set_adapters(["lightx2v", "lightx2v_2"], adapter_weights=[3., 1.])

        # Fuse LoRA weights into components
        #print("Fusing LoRA weights...")
        #self.pipe.fuse_lora(adapter_names=["lightx2v"], lora_scale=3., components=["transformer"])
        #self.pipe.fuse_lora(adapter_names=["lightx2v_2"], lora_scale=1., components=["transformer_2"])

        # Unload LoRA weights after fusing
        #self.pipe.unload_lora_weights()

        print(f"WAN VACE pipeline with LightX2V LoRA ready with variant: {variant}!")

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Process video with WAN VACE pipeline."""

        # Validate input video exists
        if not input_data.video.exists():
            raise RuntimeError(f"Input video does not exist at path: {input_data.video.path}")

        # Extract video properties and first frame
        cap = cv2.VideoCapture(input_data.video.path)

        # Get video dimensions and frame count
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise RuntimeError(f"Could not read first frame from video: {input_data.video.path}")

        # Convert BGR to RGB and create PIL Image
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        first_frame_img = PIL.Image.fromarray(frame)
        print(f"Extracted first frame from input video ({video_width}x{video_height}, {video_frame_count} frames)")

        # Use video properties instead of input parameters
        actual_width = video_width
        actual_height = video_height
        actual_num_frames = video_frame_count

        # Prepare video frames and mask using your exact function
        print(f"Preparing video with {actual_num_frames} frames at {actual_width}x{actual_height}")
        video, _ = prepare_video_and_mask(
            actual_height,
            actual_width,
            actual_num_frames,
            first_frame_img
        )

        # Process reference images if provided
        reference_images = None
        if input_data.reference_images:
            reference_images = []
            for ref_img_file in input_data.reference_images:
                if ref_img_file.exists():
                    ref_img = load_image(ref_img_file.path)
                    reference_images.append(ref_img)
            print(f"Loaded {len(reference_images)} reference images")

        # Setup generator for reproducible results
        generator = torch.Generator(device=self.device).manual_seed(input_data.seed)

        # Update boundary ratio at runtime (for dual transformer variants)
        print(f"Updating boundary ratio to: {input_data.boundary_ratio}")
        self.pipe.register_to_config(boundary_ratio=input_data.boundary_ratio)

        # Run inference with WAN VACE - using only the inputs you specified
        print(f"Generating video with prompt: {input_data.prompt}")
        output = self.pipe(
            video=video,
            prompt=input_data.prompt,
            reference_images=reference_images,
            negative_prompt=input_data.negative_prompt,
            height=actual_height,
            width=actual_width,
            num_frames=actual_num_frames,
            num_inference_steps=input_data.num_inference_steps,
            generator=generator,
        ).frames[0]

        # Export to video file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            output_path = tmp.name

        export_to_video(output, output_path, fps=16)
        print(f"Video exported to: {output_path}")

        return AppOutput(video=File(path=output_path))

    async def unload(self):
        """Clean up GPU memory and resources."""
        if hasattr(self, 'pipe'):
            del self.pipe

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        import gc
        gc.collect()
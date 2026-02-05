import os
import torch
import tempfile
import PIL.Image
from diffusers import AutoencoderKLWan, WanVACEPipeline
from diffusers.utils import export_to_video, load_image
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from typing import List, Optional
from accelerate import Accelerator

# Enable faster downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

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
    first_frame: Optional[File] = Field(None, description="Optional first frame image for image-to-video generation")
    prompt: str = Field(description="Text prompt describing the desired video content")
    reference_images: Optional[List[File]] = Field(None, description="Optional list of reference images")
    negative_prompt: Optional[str] = Field(None, description="Negative prompt to avoid unwanted content")
    height: int = Field(default=480, description="Output video height in pixels")
    width: int = Field(default=832, description="Output video width in pixels")
    num_frames: int = Field(default=45, description="Number of frames in output video")
    num_inference_steps: int = Field(default=8, ge=6, le=10, description="Number of inference steps (6-10 recommended)")
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
        variant = getattr(metadata, "app_variant", "default")
        print(f"Loading variant: {variant}")

        # Parse variant to determine configuration
        is_gguf = any(variant.startswith(prefix) for prefix in ["q4_0", "q8_0"])
        is_cpu = variant.endswith("_cpu")
        enable_offload = "_offload" in variant

        if is_gguf:
            # GGUF variants are not yet supported with the current WanVACE approach
            # Fall back to standard model for now
            print(f"GGUF variant {variant} requested, falling back to standard model...")
            model_repo = "linoyts/Wan2.2-VACE-Fun-14B-diffusers"
        else:
            # Use the standard WAN VACE model
            model_repo = "linoyts/Wan2.2-VACE-Fun-14B-diffusers"

        print(f"Loading WAN VACE pipeline from {model_repo}...")

        # Load VAE separately with float32 for stability
        print("Loading VAE...")
        self.vae = AutoencoderKLWan.from_pretrained(
            model_repo,
            subfolder="vae",
            torch_dtype=torch.float32
        )

        # Load main pipeline
        print("Loading main pipeline...")
        load_kwargs = {"vae": self.vae, "torch_dtype": torch.bfloat16}

        if variant == "low_vram" or enable_offload or is_cpu:
            print("Enabling CPU offloading...")
            self.pipe = WanVACEPipeline.from_pretrained(model_repo, **load_kwargs)
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe = WanVACEPipeline.from_pretrained(model_repo, **load_kwargs)

        if not is_cpu and not (variant == "low_vram" or enable_offload):
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
        self.pipe.set_adapters(["lightx2v", "lightx2v_2"], adapter_weights=[1., 1.])

        # Fuse LoRA weights into components
        print("Fusing LoRA weights...")
        self.pipe.fuse_lora(adapter_names=["lightx2v"], lora_scale=3., components=["transformer"])
        self.pipe.fuse_lora(adapter_names=["lightx2v_2"], lora_scale=1., components=["transformer_2"])

        # Unload LoRA weights after fusing
        self.pipe.unload_lora_weights()

        print(f"WAN VACE pipeline with LightX2V LoRA ready with variant: {variant}!")

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Process video with WAN VACE pipeline."""

        # Load first frame if provided
        first_frame_img = None
        if input_data.first_frame and input_data.first_frame.exists():
            first_frame_img = PIL.Image.open(input_data.first_frame.path).convert("RGB")
            print(f"Using provided first frame for image-to-video generation")

        # Prepare video frames and mask using your exact function
        print(f"Preparing video with {input_data.num_frames} frames at {input_data.width}x{input_data.height}")
        video, mask = prepare_video_and_mask(
            input_data.height,
            input_data.width,
            input_data.num_frames,
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

        # Run inference with WAN VACE - using only the inputs you specified
        print(f"Generating video with prompt: {input_data.prompt}")
        output = self.pipe(
            video=video,
            prompt=input_data.prompt,
            reference_images=reference_images,
            negative_prompt=input_data.negative_prompt,
            height=input_data.height,
            width=input_data.width,
            num_frames=input_data.num_frames,
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
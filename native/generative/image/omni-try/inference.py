from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from typing import Literal
import torch
import random
import numpy as np
import math
import torchvision.transforms as T
from PIL import Image
import os
import logging
from accelerate import Accelerator
from peft import LoraConfig
from safetensors import safe_open
import peft
from huggingface_hub import hf_hub_download

from .omnitry.models.transformer_flux import FluxTransformer2DModel
from .omnitry.pipelines.pipeline_flux_fill import FluxFillPipeline

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

accelerator = Accelerator()
device = accelerator.device

OBJECT_CLASSES = [
    "top clothes", "bottom clothes", "dress", "shoe", "earrings", "bracelet", 
    "necklace", "ring", "sunglasses", "glasses", "belt", "bag", "hat", "tie", "bow tie"
]

def seed_everything(seed=0):
    """Set random seed for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class AppInput(BaseAppInput):
    person_image: File = Field(description="Input image of the person")
    object_image: File = Field(description="Input image of the object to try on")
    object_class: Literal[tuple(OBJECT_CLASSES)] = Field(description="Type of object being tried on")
    steps: int = Field(default=20, ge=1, le=50, description="Number of inference steps")
    guidance_scale: float = Field(default=30.0, ge=1.0, le=50.0, description="Guidance scale for generation")
    seed: int = Field(default=-1, description="Random seed (-1 for random)")

class AppOutput(BaseAppOutput):
    image_output: File = Field(description="The generated try-on image")

def create_hacked_forward(module):
    """Create a hacked forward pass for LoRA modules."""
    def lora_forward(self, active_adapter, x, *args, **kwargs):
        result = self.base_layer(x, *args, **kwargs)
        if active_adapter is not None:
            # Store original dtype for potential future use
            # torch_result_dtype = result.dtype
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            x = x.to(lora_A.weight.dtype)
            result = result + lora_B(lora_A(dropout(x))) * scaling
        return result
    
    def hacked_lora_forward(self, x, *args, **kwargs):
        return torch.cat((
            lora_forward(self, 'vtryon_lora', x[:1], *args, **kwargs),
            lora_forward(self, 'garment_lora', x[1:], *args, **kwargs),
        ), dim=0)
    
    return hacked_lora_forward.__get__(module, type(module))

class App(BaseApp):
    async def setup(self, metadata):
        """Initialize the OmniTry model and pipeline."""
        logging.basicConfig(level=logging.INFO)
        
        # Initialize model with bfloat16 for better performance
        weight_dtype = torch.bfloat16
        
        # Load base model
        logging.info("Loading base model...")
        base_model_id = "black-forest-labs/FLUX.1-Fill-dev"
        
        # Initialize transformer with CPU offload
        self.transformer = FluxTransformer2DModel.from_pretrained(
            base_model_id,
            subfolder="transformer",
            torch_dtype=weight_dtype,
            low_cpu_mem_usage=True
        ).requires_grad_(False)
        
        # Initialize pipeline
        logging.info("Initializing pipeline...")
        self.pipeline = FluxFillPipeline.from_pretrained(
            base_model_id,
            transformer=self.transformer.eval(),
            torch_dtype=weight_dtype,
            low_cpu_mem_usage=True
        )
        
        # Enable VRAM optimizations
        self.pipeline.enable_model_cpu_offload()
        self.pipeline.vae.enable_tiling()
        
        # Configure LoRA
        logging.info("Setting up LoRA adapters...")
        lora_config = LoraConfig(
            r=16,  # LoRA rank from config
            lora_alpha=16,  # LoRA alpha from config
            init_lora_weights="gaussian",
            target_modules=[
                'x_embedder',
                'attn.to_k', 'attn.to_q', 'attn.to_v', 'attn.to_out.0', 
                'attn.add_k_proj', 'attn.add_q_proj', 'attn.add_v_proj', 'attn.to_add_out', 
                'ff.net.0.proj', 'ff.net.2', 'ff_context.net.0.proj', 'ff_context.net.2', 
                'norm1_context.linear', 'norm1.linear', 'norm.linear', 'proj_mlp', 'proj_out'
            ]
        )
        
        # Add LoRA adapters
        self.transformer.add_adapter(lora_config, adapter_name='vtryon_lora')
        self.transformer.add_adapter(lora_config, adapter_name='garment_lora')
        
        # Download and load LoRA weights
        logging.info("Loading LoRA weights...")
        lora_model_id = "Kunbyte/OmniTry"
        lora_weights_path = hf_hub_download(
            repo_id=lora_model_id,
            filename="omnitry_v1_unified.safetensors",
            resume_download=True
        )
        
        with safe_open(lora_weights_path, framework="pt") as f:
            lora_weights = {k: f.get_tensor(k) for k in f.keys()}
            self.transformer.load_state_dict(lora_weights, strict=False)
            
        # Setup object mapping from config
        self.object_map = {
            "top clothes": "replacing the top cloth",
            "bottom clothes": "replacing the bottom cloth",
            "dress": "replacing the dress",
            "shoe": "replacing the shoe",
            "earrings": "trying on earrings",
            "bracelet": "trying on bracelet",
            "necklace": "trying on necklace",
            "ring": "trying on ring",
            "sunglasses": "trying on sunglasses",
            "glasses": "trying on glasses",
            "belt": "trying on belt",
            "bag": "trying on bag",
            "hat": "trying on hat",
            "tie": "trying on tie",
            "bow tie": "trying on bow tie"
        }
        
        # Hack LoRA forward pass
        logging.info("Setting up LoRA forward pass...")
        for n, m in self.transformer.named_modules():
            if isinstance(m, peft.tuners.lora.layer.Linear):
                m.forward = create_hacked_forward(m)
        
        logging.info("Setup complete!")

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Run inference on the input images."""
        # Set random seed if provided
        if input_data.seed == -1:
            seed = random.randint(0, 2**32 - 1)
        else:
            seed = input_data.seed
        seed_everything(seed)
        
        # Load and preprocess person image
        person_image = Image.open(input_data.person_image.path)
        object_image = Image.open(input_data.object_image.path)
        
        # Calculate target size
        max_area = 1024 * 1024
        oW = person_image.width
        oH = person_image.height
        
        ratio = math.sqrt(max_area / (oW * oH))
        ratio = min(1, ratio)
        tW, tH = int(oW * ratio) // 16 * 16, int(oH * ratio) // 16 * 16
        
        # Transform person image
        transform = T.Compose([
            T.Resize((tH, tW)),
            T.ToTensor(),
        ])
        person_image = transform(person_image)
        
        # Transform and pad object image
        ratio = min(tW / object_image.width, tH / object_image.height)
        transform = T.Compose([
            T.Resize((int(object_image.height * ratio), int(object_image.width * ratio))),
            T.ToTensor(),
        ])
        object_image_padded = torch.ones_like(person_image)
        object_image = transform(object_image)
        new_h, new_w = object_image.shape[1], object_image.shape[2]
        min_x = (tW - new_w) // 2
        min_y = (tH - new_h) // 2
        object_image_padded[:, min_y: min_y + new_h, min_x: min_x + new_w] = object_image
        
        # Prepare prompts and conditions
        prompts = [self.object_map[input_data.object_class]] * 2
        img_cond = torch.stack([person_image, object_image_padded]).to(dtype=torch.bfloat16, device=device)
        mask = torch.zeros_like(img_cond).to(img_cond)
        
        # Generate image
        with torch.no_grad():
            output = self.pipeline(
                prompt=prompts,
                height=tH,
                width=tW,
                img_cond=img_cond,
                mask=mask,
                guidance_scale=input_data.guidance_scale,
                num_inference_steps=input_data.steps,
                generator=torch.Generator(device).manual_seed(seed),
            ).images[0]
        
        # Save output
        output_path = "/tmp/generated_image.png"
        output.save(output_path)
        
        return AppOutput(image_output=File(path=output_path))

    async def unload(self):
        """Clean up resources."""
        self.pipeline = None
        self.transformer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
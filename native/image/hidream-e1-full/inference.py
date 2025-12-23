import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from typing import Optional
import torch
from .pipeline_hidream_image_editing import HiDreamImageEditingPipeline
from diffusers import HiDreamImageTransformer2DModel
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast
from accelerate import Accelerator
from huggingface_hub import hf_hub_download
from diffusers import GGUFQuantizationConfig
from PIL import Image
import logging
from enum import Enum
from peft import LoraConfig
from safetensors.torch import load_file


LLAMA_MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct"
# GGUF repositories for different models
E1_GGUF_REPO_ID = "ND911/HiDream_e1_full_bf16-ggufs"  # E1 standalone models
I1_GGUF_REPO_ID = "city96/HiDream-I1-Full-gguf"       # I1 base models for LoRA

# HiDream-E1 is designed specifically for 768x768 images
FIXED_IMAGE_SIZE = 768

# Model configurations - E1 editing models
MODEL_CONFIGS = {
    # Full FP16 (I1-Full base with optional E1 LoRA)
    "full": {
        "base_repo": "HiDream-ai/HiDream-I1-Full",
        "e1_repo": "HiDream-ai/HiDream-E1-Full", 
        "guidance_scale": 5.0,
        "image_guidance_scale": 4.0,
        "num_inference_steps": 28,
        "type": "fp16"
    },
    # GGUF variants - can use I1 base + E1 LoRA or standalone E1
    "e1-q8": {
        "e1_filename": "hidream_e1_full_bf16-Q8_0.gguf",      # E1 standalone
        "i1_filename": "hidream-i1-full-Q8_0.gguf",          # I1 base for LoRA
        "guidance_scale": 5.0,
        "image_guidance_scale": 4.0,
        "num_inference_steps": 28,
        "type": "gguf"
    },
    "e1-q6k": {
        "e1_filename": "hidream_e1_full_bf16-Q6_K.gguf",
        "i1_filename": "hidream-i1-full-Q6_K.gguf",
        "guidance_scale": 5.0,
        "image_guidance_scale": 4.0,
        "num_inference_steps": 28,
        "type": "gguf"
    },
    "e1-q5k": {
        "e1_filename": "hidream_e1_full_bf16-Q5_K.gguf",
        "i1_filename": "hidream-i1-full-Q5_K.gguf",
        "guidance_scale": 5.0,
        "image_guidance_scale": 4.0,
        "num_inference_steps": 28,
        "type": "gguf"
    },
    "e1-q5km": {
        "e1_filename": "hidream_e1_full_bf16-Q5_K_M.gguf",
        "i1_filename": "hidream-i1-full-Q5_K_M.gguf",
        "guidance_scale": 5.0,
        "image_guidance_scale": 4.0,
        "num_inference_steps": 28,
        "type": "gguf"
    },
    "e1-q51": {
        "e1_filename": "hidream_e1_full_bf16-Q5_1.gguf",
        "i1_filename": "hidream-i1-full-Q5_1.gguf",
        "guidance_scale": 5.0,
        "image_guidance_scale": 4.0,
        "num_inference_steps": 28,
        "type": "gguf"
    },
    "e1-q50": {
        "e1_filename": "hidream_e1_full_bf16-Q5_0.gguf",
        "i1_filename": "hidream-i1-full-Q5_0.gguf",
        "guidance_scale": 5.0,
        "image_guidance_scale": 4.0,
        "num_inference_steps": 28,
        "type": "gguf"
    },
    "e1-q4k": {
        "e1_filename": "hidream_e1_full_bf16-Q4_K.gguf",
        "i1_filename": "hidream-i1-full-Q4_K.gguf",
        "guidance_scale": 5.0,
        "image_guidance_scale": 4.0,
        "num_inference_steps": 28,
        "type": "gguf"
    },
    "e1-q4km": {
        "e1_filename": "hidream_e1_full_bf16-Q4_K_M.gguf",
        "i1_filename": "hidream-i1-full-Q4_K_M.gguf",
        "guidance_scale": 5.0,
        "image_guidance_scale": 4.0,
        "num_inference_steps": 28,
        "type": "gguf"
    },
    "e1-q41": {
        "e1_filename": "hidream_e1_full_bf16-Q4_1.gguf",
        "i1_filename": "hidream-i1-full-Q4_1.gguf",
        "guidance_scale": 5.0,
        "image_guidance_scale": 4.0,
        "num_inference_steps": 28,
        "type": "gguf"
    },
    "e1-q40": {
        "e1_filename": "hidream_e1_full_bf16-Q4_0.gguf",
        "i1_filename": "hidream-i1-full-Q4_0.gguf",
        "guidance_scale": 5.0,
        "image_guidance_scale": 4.0,
        "num_inference_steps": 28,
        "type": "gguf"
    },
    "e1-q2k": {
        "e1_filename": "hidream_e1_full_bf16-Q2_K.gguf",
        "i1_filename": "hidream-i1-full-Q2_K.gguf",
        "guidance_scale": 5.0,
        "image_guidance_scale": 4.0,
        "num_inference_steps": 28,
        "type": "gguf"
    },
}

class AppInput(BaseAppInput):
    prompt: str = Field(description="Editing instruction followed by target description. Format: 'Editing Instruction: {instruction}. Target Image Description: {description}'")
    image: File = Field(description="Input image to edit")
    negative_prompt: str = Field(default="low resolution, blur", description="Negative prompt to avoid unwanted features")
    seed: Optional[int] = Field(default=-1, description="The seed for the random number generator (-1 for random)")
    
    # Generation parameters with E1 defaults
    num_inference_steps: int = Field(
        default=28, 
        ge=1, 
        le=100, 
        description="Number of denoising steps"
    )
    guidance_scale: float = Field(
        default=5.0,
        ge=0.0,
        le=20.0,
        description="Text guidance scale - how closely to follow the prompt"
    )
    image_guidance_scale: float = Field(
        default=4.0,
        ge=0.0,
        le=20.0,
        description="Image guidance scale - how closely to follow the input image"
    )
    refine_strength: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Refinement strength (0.0 disables, 1.0 full refinement). Works with all variants - uses I1 base + E1 LoRA when > 0.0, standalone E1 when = 0.0."
    )

class AppOutput(BaseAppOutput):
    result: File

class App(BaseApp):
    def _setup_lora_refinement(self, transformer):
        """Setup LoRA refinement (applies to both FP16 and GGUF)"""
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["to_k", "to_q", "to_v", "to_out", "to_k_t", "to_q_t", "to_v_t", "to_out_t", "w1", "w2", "w3", "final_layer.linear"],
            init_lora_weights="gaussian",
        )
        transformer.add_adapter(lora_config)
        # max_seq is set in the caller
        lora_ckpt_path = hf_hub_download(repo_id="HiDream-ai/HiDream-E1-Full", filename="HiDream-E1-Full.safetensors")
        lora_ckpt = load_file(lora_ckpt_path, device=str(self.accelerator.device))
        src_state_dict = transformer.state_dict()
        reload_keys = [k for k in lora_ckpt if "lora" not in k]
        self.reload_keys = {
            "editing": {k: v for k, v in lora_ckpt.items() if k in reload_keys},
            "refine": {k: v for k, v in src_state_dict.items() if k in reload_keys},
        }
        info = transformer.load_state_dict(lora_ckpt, strict=False)
        assert len(info.unexpected_keys) == 0
        return transformer

    def _should_use_refinement_architecture(self, refine_strength):
        """Determine if we should use I1 base + E1 LoRA architecture based on refine_strength"""
        # GGUF models don't support LoRA, so always use standalone E1 architecture
        if self.config["type"] == "gguf":
            if refine_strength > 0.0:
                print(f"[DEBUG] GGUF models don't support refinement - forcing refine_strength to 0.0")
            return False
        # For FP16 models, use refinement architecture when requested
        return refine_strength > 0.0

    def _setup_pipeline_for_refinement(self, use_refinement):
        """Set up pipeline architecture based on whether refinement is needed"""
        if use_refinement:
            # Use I1 base + E1 LoRA architecture
            print(f"[DEBUG] Setting up I1 base + E1 LoRA architecture for refinement")
            
            if self.config["type"] == "fp16":
                # Load I1-Full transformer (FP16)
                transformer = HiDreamImageTransformer2DModel.from_pretrained(
                    "HiDream-ai/HiDream-I1-Full",
                    subfolder="transformer"
                )
                print(f"[DEBUG] Loaded I1-Full transformer for FP16")
                transformer.max_seq = 4608
            else:
                # Load I1 GGUF transformer (reference logic)
                i1_filename = self.config["i1_filename"]
                print(f"[DEBUG] Downloading I1 GGUF: {i1_filename} from {I1_GGUF_REPO_ID}...")
                ckpt_path = hf_hub_download(repo_id=I1_GGUF_REPO_ID, filename=i1_filename)
                transformer = HiDreamImageTransformer2DModel.from_single_file(
                    ckpt_path,
                    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
                    torch_dtype=torch.bfloat16
                )
                print(f"[DEBUG] Loaded I1 GGUF transformer: {i1_filename}")
                transformer.max_seq = 4608
            
            # Apply E1 LoRA for both FP16 and GGUF
            transformer = self._setup_lora_refinement(transformer)
            print(f"[DEBUG] Applied E1 LoRA to transformer (type: {self.config['type']})")
            
            # Always use HiDream-ai/HiDream-I1-Full as pipeline base
            pipeline_base = "HiDream-ai/HiDream-I1-Full"
        else:
            # Use standalone E1 architecture (no refinement)
            print(f"[DEBUG] Setting up standalone E1 architecture (no refinement)")
            
            if self.config["type"] == "fp16":
                # Load E1-Full transformer directly
                transformer = HiDreamImageTransformer2DModel.from_pretrained(
                    self.config["e1_repo"], 
                    subfolder="transformer"
                )
                print(f"[DEBUG] Loaded E1-Full transformer for FP16")
                pipeline_base = self.config["e1_repo"]
            else:
                # Load E1 GGUF transformer
                e1_filename = self.config["e1_filename"]
                print(f"[DEBUG] Downloading E1 GGUF: {e1_filename} from {E1_GGUF_REPO_ID}...")
                ckpt_path = hf_hub_download(repo_id=E1_GGUF_REPO_ID, filename=e1_filename)
                transformer = HiDreamImageTransformer2DModel.from_single_file(
                    ckpt_path,
                    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
                    torch_dtype=torch.bfloat16
                )
                print(f"[DEBUG] Loaded E1 GGUF transformer: {e1_filename}")
                pipeline_base = "HiDream-ai/HiDream-E1-Full"
            
            # Set sequence length for standalone E1 (both FP16 and GGUF)
            transformer.max_seq = 4608
        
        # Create the pipeline
        self.pipe = HiDreamImageEditingPipeline.from_pretrained(
            pipeline_base,
            tokenizer_4=self.tokenizer,
            text_encoder_4=self.text_encoder,
            transformer=transformer,
            torch_dtype=torch.bfloat16
        )
        
        # Store transformer reference
        self.transformer = transformer
        self.lora_setup_done = True
        self.pipeline_setup_done = True
        
        print(f"[DEBUG] Pipeline setup completed with base: {pipeline_base}")

    def _ensure_lora_setup(self):
        """This method is no longer used - pipeline setup is done dynamically"""
        pass

    async def setup(self, metadata):
        """Initialize your model and resources here."""
        self.accelerator = Accelerator()
        print(f"[DEBUG] torch.cuda.is_available(): {torch.cuda.is_available()}")
        print(f"[DEBUG] Accelerator device: {self.accelerator.device}")
        
        # Set up variant and model type
        self.variant = getattr(metadata, "app_variant", "default")
        
        # Map variants to model configs
        variant_mapping = {
            "default": "e1-q8",
            # E1 GGUF variants
            "e1-q6k": "e1-q6k", "e1-q5k": "e1-q5k", "e1-q5km": "e1-q5km",
            "e1-q51": "e1-q51", "e1-q50": "e1-q50", "e1-q4k": "e1-q4k",
            "e1-q4km": "e1-q4km", "e1-q41": "e1-q41", "e1-q40": "e1-q40",
            "e1-q2k": "e1-q2k",
            # Full FP16
            "full": "full"
        }
        
        self.model = variant_mapping.get(self.variant, "e1-q8")
        self.config = MODEL_CONFIGS[self.model]
        
        print(f"[DEBUG] Using model variant: {self.variant} -> {self.model}")
        print(f"[DEBUG] Model type: {self.config['type']}")
        
        # Load tokenizer and text encoder
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(LLAMA_MODEL_NAME)
        self.text_encoder = LlamaForCausalLM.from_pretrained(
            LLAMA_MODEL_NAME,
            output_hidden_states=True,
            output_attentions=True,
            torch_dtype=torch.bfloat16,
        )

        # Initialize attributes
        self.reload_keys = None
        self.transformer = None
        self.lora_setup_done = True  # Will be set to False if LoRA setup is needed
        self.pipeline_setup_done = False
        
        print(f"[DEBUG] Initial setup for model type: {self.config['type']}")
        
        # Store config for later dynamic setup based on refine_strength
        self.setup_config = self.config
        
        print(f"[DEBUG] Setup completed - pipeline will be created dynamically based on refine_strength")
        self.device = self.accelerator.device

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Run image editing on the input data."""
        # Load and prepare input image
        input_image = Image.open(input_data.image.path).convert('RGB')
        original_width, original_height = input_image.size
        
        # Resize to 768x768 like reference (this fixes the tensor size issues)
        input_image = input_image.resize((768, 768))
        
        # Optional: Refine instruction if module is available
        try:
            from .instruction_refinement import refine_instruction
            prompt = refine_instruction(src_image=input_image, src_instruction=input_data.prompt)
            print(f"[DEBUG] Original prompt: {input_data.prompt}")
            print(f"[DEBUG] Refined prompt: {prompt}")
        except ImportError:
            prompt = input_data.prompt
            print(f"[DEBUG] Using original prompt (refinement not available): {prompt}")
        
        # Set up generator for reproducibility
        seed = input_data.seed if input_data.seed is not None and input_data.seed != -1 else torch.randint(0, 1000000, (1,), device=self.accelerator.device).item()
        generator = torch.Generator(self.accelerator.device).manual_seed(seed)

        print(f"[DEBUG] Using parameters: steps={input_data.num_inference_steps}, guidance={input_data.guidance_scale}, image_guidance={input_data.image_guidance_scale}, refine_strength={input_data.refine_strength}")

        # All variants now support refinement via I1 base + E1 LoRA architecture
        effective_refine_strength = input_data.refine_strength
                    
        print(f"[DEBUG] Effective refine_strength: {effective_refine_strength} (model type: {self.config['type']})")

        # Set up pipeline architecture based on refinement needs
        use_refinement = effective_refine_strength > 0.0
        print(f"[DEBUG] Will use refinement architecture: {use_refinement} (refine_strength > 0.0: {effective_refine_strength > 0.0})")
        if not self.pipeline_setup_done:
            self._setup_pipeline_for_refinement(use_refinement)
            
            # Device setup after pipeline creation
            if self.config["type"] == "gguf":
                self.pipe.enable_model_cpu_offload()
            else:
                self.pipe = self.pipe.to(self.accelerator.device, torch.bfloat16)
            
            print(f"[DEBUG] Pipeline moved to device: {self.accelerator.device}")
        
        # Generate edited image (following reference implementation exactly)
        images = self.pipe(
            prompt=prompt,
            negative_prompt=input_data.negative_prompt,
            image=input_image,
            guidance_scale=input_data.guidance_scale,
            image_guidance_scale=input_data.image_guidance_scale,
            num_inference_steps=input_data.num_inference_steps,
            generator=generator,
            refine_strength=effective_refine_strength,
            reload_keys=self.reload_keys  # Pass the LoRA keys for refinement (only for FP16)
        ).images

        # Resize output back to original dimensions (like reference)
        output_image = images[0].resize((original_width, original_height))
        output_path = "/tmp/edited_image.png"
        output_image.save(output_path)
        
        return AppOutput(result=File(path=output_path))

    async def unload(self):
        """Clean up resources here."""
        del self.pipe
        del self.text_encoder
        del self.tokenizer
        torch.cuda.empty_cache() 
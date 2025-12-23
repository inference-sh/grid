import os
import torch
from huggingface_hub import snapshot_download
from inferencesh import BaseApp, BaseAppOutput, File
from inferencesh.models.llm import (
    LLMInput,
    ImageCapabilityMixin,
)
from pydantic import Field
from typing import Optional, AsyncGenerator
from PIL import Image
from copy import deepcopy
import asyncio
from tqdm import tqdm

import sys
# Add both current directory and Bagel directory to path
# Current dir: for "from Bagel.xxx" imports to work
# Bagel dir: for Bagel's internal "from data.xxx" imports to work
current_dir = os.path.dirname(os.path.abspath(__file__))
bagel_dir = os.path.join(current_dir, "Bagel")
sys.path.append(current_dir)
sys.path.append(bagel_dir)

# Import from local Bagel folder
from Bagel.data.transforms import ImageTransform
from Bagel.data.data_utils import pil_img2rgb, add_special_tokens
from Bagel.modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from Bagel.modeling.qwen2 import Qwen2Tokenizer
from Bagel.modeling.autoencoder import load_ae
from Bagel.inferencer import InterleaveInferencer
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights


class AppInput(LLMInput, ImageCapabilityMixin):
    thinking_enabled: bool = Field(default=False, description="Enable thinking mode for enhanced reasoning")
    analysis_mode: bool = Field(default=False, description="Enable understanding mode for image analysis only")


class AppOutput(BaseAppOutput):
    response: Optional[str] = Field(None, description="Generated text response (thinking or understanding)")
    reasoning: Optional[str] = Field(None, description="Thinking content")
    image: Optional[File] = Field(None, description="Generated or processed image")


def setup_model_configs(model_path: str):
    """Setup model configurations for BAGEL."""
    # LLM config preparing
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    # ViT config preparing
    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    # VAE loading
    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

    # Bagel config preparing
    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config, 
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=64,
    )

    return llm_config, vit_config, vae_model, vae_config, config


def setup_device_map(model):
    """Setup device mapping for 80GB A100."""
    max_mem_per_gpu = "80GiB"  # Using 80GB A100

    device_map = infer_auto_device_map(
        model,
        max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )

    same_device_modules = [
        'language_model.model.embed_tokens',
        'time_embedder',
        'latent_pos_embed',
        'vae2llm',
        'llm2vae',
        'connector',
        'vit_pos_embed'
    ]

    if torch.cuda.device_count() == 1:
        first_device = device_map.get(same_device_modules[0], "cuda:0")
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device
            else:
                device_map[k] = "cuda:0"
    else:
        first_device = device_map.get(same_device_modules[0])
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device

    return device_map


def get_inference_hyperparams(thinking_enabled: bool, analysis_mode: bool, has_input_image: bool):
    """Get inference hyperparameters based on mode."""
    if analysis_mode:
        # Understanding mode - text only output
        return {
            "max_think_token_n": 1000,
            "do_sample": False,
            "understanding_output": True,
        }
    elif has_input_image:
        # Editing mode
        if thinking_enabled:
            return {
                "max_think_token_n": 1000,
                "do_sample": False,
                "think": True,
                "cfg_text_scale": 4.0,
                "cfg_img_scale": 2.0,
                "cfg_interval": [0.0, 1.0],
                "timestep_shift": 3.0,
                "num_timesteps": 50,
                "cfg_renorm_min": 0.0,
                "cfg_renorm_type": "text_channel",
            }
        else:
            return {
                "cfg_text_scale": 4.0,
                "cfg_img_scale": 2.0,
                "cfg_interval": [0.0, 1.0],
                "timestep_shift": 3.0,
                "num_timesteps": 50,
                "cfg_renorm_min": 1.0,
                "cfg_renorm_type": "text_channel",
            }
    else:
        # Generation mode
        if thinking_enabled:
            return {
                "max_think_token_n": 1000,
                "do_sample": False,
                "think": True,
                "cfg_text_scale": 4.0,
                "cfg_img_scale": 1.0,
                "cfg_interval": [0.4, 1.0],
                "timestep_shift": 3.0,
                "num_timesteps": 50,
                "cfg_renorm_min": 1.0,
                "cfg_renorm_type": "global",
            }
        else:
            return {
                "cfg_text_scale": 4.0,
                "cfg_img_scale": 1.0,
                "cfg_interval": [0.4, 1.0],
                "timestep_shift": 3.0,
                "num_timesteps": 50,
                "cfg_renorm_min": 1.0,
                "cfg_renorm_type": "global",
            }


class App(BaseApp):
    async def setup(self, metadata):
        """Initialize the BAGEL model and resources."""
        print("🚀 Setting up BAGEL model...")
        
        # Download model from HuggingFace using cache
        repo_id = "ByteDance-Seed/BAGEL-7B-MoT"
        print("📥 Downloading model from HuggingFace...")
        cache_dir = snapshot_download(
            repo_id=repo_id,
            resume_download=True,
            allow_patterns=["*.json", "*.safetensors", "*.bin", "*.py", "*.md", "*.txt"],
        )
        
        model_path = cache_dir
        print(f"✅ Model downloaded to: {model_path}")

        # Setup model configurations
        print("⚙️ Setting up model configurations...")
        llm_config, vit_config, vae_model, vae_config, config = setup_model_configs(model_path)
        
        # Initialize empty model
        print("🏗️ Initializing model architecture...")
        with init_empty_weights():
            language_model = Qwen2ForCausalLM(llm_config)
            vit_model = SiglipVisionModel(vit_config)
            model = Bagel(language_model, vit_model, config)
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

        # Setup tokenizer
        print("📝 Setting up tokenizer...")
        tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
        tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

        # Setup image transforms
        print("🖼️ Setting up image transforms...")
        vae_transform = ImageTransform(1024, 512, 16)
        vit_transform = ImageTransform(980, 224, 14)

        # Setup device mapping and load model
        device_map = setup_device_map(model)
        print(f"🎯 Device map: {device_map}")

        print("⏳ Loading model weights (this may take a while)...")
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=os.path.join(model_path, "ema.safetensors"),
            device_map=device_map,
            offload_buffers=True,
            dtype=torch.bfloat16,
            force_hooks=True,
            offload_folder="/tmp/offload"
        )
        
        model = model.eval()
        print("✅ Model loaded successfully")

        # Initialize inferencer
        print("🎭 Initializing inferencer...")
        self.inferencer = InterleaveInferencer(
            model=model, 
            vae_model=vae_model, 
            tokenizer=tokenizer, 
            vae_transform=vae_transform, 
            vit_transform=vit_transform, 
            new_token_ids=new_token_ids
        )
        
        print("🎉 BAGEL setup complete!")

    async def run(self, input_data: AppInput, metadata) -> AsyncGenerator[AppOutput, None]:
        """Run BAGEL inference on the input data with streaming support."""
        
        # Determine input image
        input_image = None
        if input_data.image and input_data.image.exists():
            input_image = Image.open(input_data.image.path).convert("RGB")
        
        # Get inference hyperparameters based on mode
        inference_params = get_inference_hyperparams(
            thinking_enabled=input_data.thinking_enabled,
            analysis_mode=input_data.analysis_mode,
            has_input_image=input_image is not None
        )
        
        # For streaming modes (thinking or understanding), we need to implement custom logic
        if input_data.thinking_enabled or input_data.analysis_mode:
            # Stream the text generation process
            async for output in self._stream_inference(input_image, input_data.text, inference_params):
                yield output
        else:
            # Non-streaming mode - run regular inference and yield final result
            output_dict = self.inferencer(
                image=input_image,
                text=input_data.text,
                **inference_params
            )
            
            # Prepare output - map to correct fields based on mode
            if input_data.analysis_mode:
                # Understanding output goes to response field
                yield AppOutput(response=output_dict.get('text'))
            else:
                # Regular generation - check if thinking was involved
                output_text = output_dict.get('text')
                result_text = output_text.replace("</think>", "").replace("<think>", "").replace("<|im_end|>", "") if output_text else None
                result_image = output_dict.get('image')
                
                # Save image if generated
                output_image_file = None
                if result_image is not None:
                    output_path = "/tmp/generated_image.png"
                    result_image.save(output_path)
                    output_image_file = File(path=output_path)
                
                # If thinking was enabled, result goes to reasoning, otherwise no text output
                if input_data.thinking_enabled and result_text:
                    yield AppOutput(
                        reasoning=result_text,
                        image=output_image_file
                    )
                else:
                    yield AppOutput(image=output_image_file)

    async def _stream_inference(self, input_image, text, inference_params):
        """Stream text generation for thinking and understanding modes using real token-by-token streaming."""
        
        # Prepare input list
        input_list = []
        if input_image is not None:
            input_list.append(input_image)
        if text is not None:
            input_list.append(text)
        
        # Determine the mode for progress messages
        is_understanding = inference_params.get('understanding_output', False)
        is_thinking = inference_params.get('think', False)
        
        if is_understanding:
            print("🧠 Analyzing image...")
        elif is_thinking:
            print("💭 Thinking about the request...")
        
        # Use the streaming inferencer
        final_reasoning = None
        for stream_output in self.inferencer.interleave_inference_streaming(input_list, **inference_params):
            stream_type = stream_output["type"]
            content = stream_output["content"]
            
            if stream_type == "text_token":
                # Understanding mode - yield incremental response
                accumulated = stream_output["accumulated"]
                yield AppOutput(response=accumulated)
                
            elif stream_type == "thinking_token":
                # Thinking mode - yield incremental thinking content
                accumulated = stream_output["accumulated"]
                accumulated = accumulated.replace("</think>", "").replace("<think>", "").replace("<|im_end|>", "") if accumulated else None
                final_reasoning = accumulated
                yield AppOutput(reasoning=accumulated)
                
            elif stream_type == "final_text":
                # Final understanding response
                print("✅ Analysis complete!")
                yield AppOutput(response=content)
                
            elif stream_type == "final_thinking":
                # Final thinking content (without image yet)
                content_cleaned = content.replace("</think>", "").replace("<think>", "").replace("<|im_end|>", "") if content else None
                final_reasoning = content_cleaned
                print("✅ Thinking complete! Now generating image...")
                yield AppOutput(reasoning=final_reasoning)
                
            elif stream_type == "final_image":
                # Final image result - save and yield
                output_path = "/tmp/generated_image.png"
                content.save(output_path)
                output_image_file = File(path=output_path)
                
                # For thinking mode, include both final thinking and image
                if inference_params.get('think', False):
                    # Get the final thinking content from previous yield
                    # This will be the complete result
                    yield AppOutput(
                        reasoning=final_reasoning,  # Don't repeat thinking content
                        image=output_image_file
                    )
                else:
                    # Pure image generation
                    yield AppOutput(image=output_image_file)

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
import os
import sys
import torch
import tempfile
import base64
import json
from typing import Optional, List, Union, Dict
from pathlib import Path
import imageio
import numpy as np
from PIL import Image
from pydantic import BaseModel
from huggingface_hub import hf_hub_download

# Add LTX-Video to path
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "ltx"))

# Import from LTX-Video
from ltx_video.pipelines.pipeline_ltx_video import ConditioningItem
from ltx.inference import (
    create_ltx_video_pipeline,
    load_image_to_tensor_with_resize_and_crop,
    calculate_padding,
    get_device,
    prepare_conditioning
)

class ConditioningImage(BaseModel):
    image: str  # Base64 encoded image
    frame_index: Optional[int] = None  # If None, will be automatically assigned
    strength: float = 1.0  # Default to maximum strength

class AppInput(BaseAppInput):
    prompt: str
    negative_prompt: Optional[str] = "worst quality, inconsistent motion, blurry, jittery, distorted"
    height: Optional[int] = 480
    width: Optional[int] = 704
    num_frames: Optional[int] = 121
    frame_rate: Optional[int] = 25
    num_inference_steps: Optional[int] = 40
    guidance_scale: Optional[float] = 3.0
    seed: Optional[int] = 171198
    conditioning_images: Optional[List[ConditioningImage]] = None  # List of conditioning images with their parameters

class AppOutput(BaseAppOutput):
    video: File

class App(BaseApp):
    async def setup(self):
        """Initialize the LTX-Video model."""
        self.ckpt_path = hf_hub_download(repo_id="Lightricks/LTX-Video", filename="ltx-video-2b-v0.9.5.safetensors", repo_type='model')

        self.device = get_device()
        
        self.pipeline = create_ltx_video_pipeline(
            ckpt_path=self.ckpt_path,
            precision="bfloat16",
            text_encoder_model_name_or_path="PixArt-alpha/PixArt-XL-2-1024-MS",
            device=self.device,
            enhance_prompt=True,
            prompt_enhancer_image_caption_model_name_or_path="MiaoshouAI/Florence-2-large-PromptGen-v2.0",
            prompt_enhancer_llm_model_name_or_path="unsloth/Llama-3.2-3B-Instruct",
        )

    async def run(self, input_data: AppInput) -> AppOutput:
        """Run video generation with LTX-Video."""
        # Set up generator with seed
        generator = torch.Generator(device=self.device).manual_seed(input_data.seed)
        
        # Adjust dimensions to be divisible by 32 and num_frames to be (N * 8 + 1)
        height_padded = ((input_data.height - 1) // 32 + 1) * 32
        width_padded = ((input_data.width - 1) // 32 + 1) * 32
        num_frames_padded = ((input_data.num_frames - 2) // 8 + 1) * 8 + 1
        
        padding = calculate_padding(
            input_data.height, 
            input_data.width, 
            height_padded, 
            width_padded
        )
        
        # Prepare conditioning items
        conditioning_items = []
        
        # Handle conditioning images if provided
        if input_data.conditioning_images and len(input_data.conditioning_images) > 0:
            # Collect all images with explicit frame indices
            explicit_frames = []
            unassigned_images = []
            
            for item in input_data.conditioning_images:
                if item.frame_index is not None:
                    explicit_frames.append(item)
                else:
                    unassigned_images.append(item)
            
            # Assign frame indices to images without explicit indices
            num_unassigned = len(unassigned_images)
            if num_unassigned > 0:
                # Get list of already assigned frames
                assigned_frames = [item.frame_index for item in explicit_frames]
                
                if num_unassigned == 1:
                    # If there's only one unassigned image, place it at the middle frame
                    # or at 0 if no explicit frames yet
                    if not assigned_frames:
                        unassigned_images[0].frame_index = 0
                    else:
                        mid_frame = input_data.num_frames // 2
                        # Try to avoid collision with explicit frames
                        while mid_frame in assigned_frames:
                            mid_frame = (mid_frame + 1) % input_data.num_frames
                        unassigned_images[0].frame_index = mid_frame
                else:
                    # For multiple images, distribute them evenly across available frames
                    available_frames = [i for i in range(input_data.num_frames) if i not in assigned_frames]
                    
                    # If we have fewer available frames than images, we'll have to reuse some frames
                    if len(available_frames) < num_unassigned:
                        available_frames = list(range(input_data.num_frames))
                    
                    # Select frames at regular intervals from available frames
                    step = len(available_frames) / num_unassigned
                    for i, item in enumerate(unassigned_images):
                        frame_idx = available_frames[int(i * step)]
                        item.frame_index = frame_idx
            
            # Now process all images (both explicit and assigned)
            all_images = explicit_frames + unassigned_images
            
            # Process each conditioning image
            for item in all_images:
                # Decode base64 image
                image_data = base64.b64decode(item.image)
                temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                temp_file.write(image_data)
                temp_file.close()
                
                # Create conditioning item
                frame_tensor = load_image_to_tensor_with_resize_and_crop(
                    temp_file.name, input_data.height, input_data.width
                )
                frame_tensor = torch.nn.functional.pad(frame_tensor, padding)
                
                # Ensure frame index is within valid range
                frame_index = max(0, min(item.frame_index, input_data.num_frames - 1))
                
                conditioning_items.append(ConditioningItem(frame_tensor, frame_index, item.strength))
                
                # Clean up temp file
                os.unlink(temp_file.name)
        
        # Prepare input for the pipeline
        sample = {
            "prompt": input_data.prompt,
            "prompt_attention_mask": None,
            "negative_prompt": input_data.negative_prompt,
            "negative_prompt_attention_mask": None,
        }
        
        # Run inference
        images = self.pipeline(
            num_inference_steps=input_data.num_inference_steps,
            num_images_per_prompt=1,
            guidance_scale=input_data.guidance_scale,
            generator=generator,
            output_type="pt",
            height=height_padded,
            width=width_padded,
            num_frames=num_frames_padded,
            frame_rate=input_data.frame_rate,
            **sample,
            conditioning_items=conditioning_items if conditioning_items else None,
            is_video=True,
            vae_per_channel_normalize=True,
            enhance_prompt=True,
        ).images
        
        # Crop the padded images to the desired resolution and number of frames
        (pad_left, pad_right, pad_top, pad_bottom) = padding
        pad_bottom = -pad_bottom if pad_bottom != 0 else images.shape[3]
        pad_right = -pad_right if pad_right != 0 else images.shape[4]
        images = images[:, :, :input_data.num_frames, pad_top:pad_bottom, pad_left:pad_right]
        
        # Convert to video
        video_np = images[0].permute(1, 2, 3, 0).cpu().float().numpy()
        video_np = (video_np * 255).astype(np.uint8)
        
        # Save video to temporary file
        output_path = "/tmp/output_video.mp4"
        with imageio.get_writer(output_path, fps=input_data.frame_rate) as video:
            for frame in video_np:
                video.append_data(frame)
        
        return AppOutput(video=File(path=output_path))

    async def unload(self):
        """Clean up resources."""
        # Free up GPU memory
        if hasattr(self, 'pipeline'):
            del self.pipeline
        torch.cuda.empty_cache()
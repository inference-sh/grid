from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from typing import Optional
import torch
from diffusers import WanPipeline, AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from transformers import CLIPVisionModel
import numpy as np
from huggingface_hub import hf_hub_download

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

i2v_effects = {
    "Assassin": {
        "repo": "inference-sh/remade-ai-wan2.1-i2v-effects",
        "folder": "Assassin",
        "file": "assassin_45_epochs.safetensors",
    },
    "Baby": {
        "repo": "inference-sh/remade-ai-wan2.1-i2v-effects",
        "folder": "Baby",
        "file": "baby_50_epochs.safetensors",
    },
    "Bride": {
        "repo": "inference-sh/remade-ai-wan2.1-i2v-effects",
        "folder": "Bride",
        "file": "bride_50_epochs.safetensors",
    },
    "Cakeify": {
        "repo": "inference-sh/remade-ai-wan2.1-i2v-effects",
        "folder": "Cakeify",
        "file": "cakeify_16_epochs.safetensors",
    },
    "Cartoon_Jaw_Drop": {
        "repo": "inference-sh/remade-ai-wan2.1-i2v-effects",
        "folder": "Cartoon_Jaw_Drop",
        "file": "cartoon_jaw_drop_50_epochs.safetensors",
    },
    "Classy": {
        "repo": "inference-sh/remade-ai-wan2.1-i2v-effects",
        "folder": "Classy",
        "file": "classy_45_epochs.safetensors",
    },
    "Crush": {
        "repo": "inference-sh/remade-ai-wan2.1-i2v-effects",
        "folder": "Crush",
        "file": "crush_20_epochs.safetensors",
    },
    "Deflate": {
        "repo": "inference-sh/remade-ai-wan2.1-i2v-effects",
        "folder": "Deflate",
        "file": "deflate_20_epochs.safetensors",
    },
    "Disney-Princess": {
        "repo": "inference-sh/remade-ai-wan2.1-i2v-effects",
        "folder": "Disney-Princess",
        "file": "disney_princess_45_epochs.safetensors",
    },
    "Dolly-Effect": {
        "repo": "inference-sh/remade-ai-wan2.1-i2v-effects",
        "folder": "Dolly-Effect",
        "file": "dolly_25_epochs.safetensors",
    },
    "Electrify": {
        "repo": "inference-sh/remade-ai-wan2.1-i2v-effects",
        "folder": "Electrify",
        "file": "electrify_50_epochs.safetensors",
    },
    "Explode": {
        "repo": "inference-sh/remade-ai-wan2.1-i2v-effects",
        "folder": "Explode",
        "file": "explode_30_epochs.safetensors",
    },
    "Fus-Ro-Dah": {
        "repo": "inference-sh/remade-ai-wan2.1-i2v-effects",
        "folder": "Fus-Ro-Dah",
        "file": "fus_ro_dah_20_epochs.safetensors",
    },
    "Gun-Shooting": {
        "repo": "inference-sh/remade-ai-wan2.1-i2v-effects",
        "folder": "Gun-Shooting",
        "file": "gun_20_epochs.safetensors",
    },
    "Hug-Jesus": {
        "repo": "inference-sh/remade-ai-wan2.1-i2v-effects",
        "folder": "Hug-Jesus",
        "file": "hug_jesus_20_epochs.safetensors",
    },
    "Hulk-Transformation": {
        "repo": "inference-sh/remade-ai-wan2.1-i2v-effects",
        "folder": "Hulk-Transformation",
        "file": "hulk_35_epochs.safetensors",
    },
    "Inflate": {
        "repo": "inference-sh/remade-ai-wan2.1-i2v-effects",
        "folder": "Inflate",
        "file": "inflate_20_epochs.safetensors",
    },
    "Jumpscare": {
        "repo": "inference-sh/remade-ai-wan2.1-i2v-effects",
        "folder": "Jumpscare",
        "file": "jumpscare_35_epochs.safetensors",
    },
    "Jungle": {
        "repo": "inference-sh/remade-ai-wan2.1-i2v-effects",
        "folder": "Jungle",
        "file": "jungle_50_epochs.safetensors",
    },
    "Laughing": {
        "repo": "inference-sh/remade-ai-wan2.1-i2v-effects",
        "folder": "Laughing",
        "file": "laughing_15_epochs.safetensors",
    },
    "Mona-Lisa": {
        "repo": "inference-sh/remade-ai-wan2.1-i2v-effects",
        "folder": "Mona-Lisa",
        "file": "mona_lisa_45_epochs.safetensors",
    },
    "Muscle": {
        "repo": "inference-sh/remade-ai-wan2.1-i2v-effects",
        "folder": "Muscle",
        "file": "muscle_18_epochs.safetensors",
    },
    "Painting": {
        "repo": "inference-sh/remade-ai-wan2.1-i2v-effects",
        "folder": "Painting",
        "file": "painting_50_epochs.safetensors",
    },
    "Pirate-Captain": {
        "repo": "inference-sh/remade-ai-wan2.1-i2v-effects",
        "folder": "Pirate-Captain",
        "file": "pirate_captain_50_epochs.safetensors",
    },
    "Princess": {
        "repo": "inference-sh/remade-ai-wan2.1-i2v-effects",
        "folder": "Princess",
        "file": "princess_45_epochs.safetensors",
    },
    "Puppy": {
        "repo": "inference-sh/remade-ai-wan2.1-i2v-effects",
        "folder": "Puppy",
        "file": "puppy_50_epochs.safetensors",
    },
    "Robot-Face-Reveal": {
        "repo": "inference-sh/remade-ai-wan2.1-i2v-effects",
        "folder": "Robot-Face-Reveal",
        "file": "robot_face_reveal_35_epochs.safetensors",
    },
    "Rotate": {
        "repo": "inference-sh/remade-ai-wan2.1-i2v-effects",
        "folder": "Rotate",
        "file": "rotate_20_epochs.safetensors",
    },
    "Samurai": {
        "repo": "inference-sh/remade-ai-wan2.1-i2v-effects",
        "folder": "Samurai",
        "file": "samurai_50_epochs.safetensors",
    },
    "Selfie-With-Younger-Self": {
        "repo": "inference-sh/remade-ai-wan2.1-i2v-effects",
        "folder": "Selfie-With-Younger-Self",
        "file": "selfie_younger_self_15_epochs.safetensors",
    },
    "Snow-White": {
        "repo": "inference-sh/remade-ai-wan2.1-i2v-effects",
        "folder": "Snow-White",
        "file": "snow_white_50_epochs.safetensors",
    },
    "Squish": {
        "repo": "inference-sh/remade-ai-wan2.1-i2v-effects",
        "folder": "Squish",
        "file": "squish_18.safetensors",
    },
    "Super-Saiyan": {
        "repo": "inference-sh/remade-ai-wan2.1-i2v-effects",
        "folder": "Super-Saiyan",
        "file": "super_saiyan_35_epochs.safetensors",
    },
    "VIP": {
        "repo": "inference-sh/remade-ai-wan2.1-i2v-effects",
        "folder": "VIP",
        "file": "vip_50_epochs.safetensors",
    },
    "Warrior": {
        "repo": "inference-sh/remade-ai-wan2.1-i2v-effects",
        "folder": "Warrior",
        "file": "warrior_45_epochs.safetensors",
    },
    "Zen": {
        "repo": "inference-sh/remade-ai-wan2.1-i2v-effects",
        "folder": "Zen",
        "file": "zen_50_epochs.safetensors",
    },
    "angry-face": {
        "repo": "inference-sh/remade-ai-wan2.1-i2v-effects",
        "folder": "angry-face",
        "file": "angry_face_5_epochs.safetensors",
    },
    "crying": {
        "repo": "inference-sh/remade-ai-wan2.1-i2v-effects",
        "folder": "crying",
        "file": "crying_20_epochs.safetensors",
    },
    "kissing": {
        "repo": "inference-sh/remade-ai-wan2.1-i2v-effects",
        "folder": "kissing",
        "file": "kissing_30_epochs.safetensors",
    },
}


class AppInput(BaseAppInput):
    prompt: str = Field(description="The input prompt describing the video to generate")
    image: File = Field(description="Optional input image for I2V mode")
    # lora_path: Optional[str] = Field(None, description="Path to LoRA weights File")
    i2v_effect: Optional[str] = Field(description="Choose an effect", enum=list(i2v_effects.keys()))
    negative_prompt: str = Field(
        description="The negative prompt to guide generation", default=""
    )
    num_frames: int = Field(description="Number of frames to generate", default=33)
    fps: int = Field(description="Frames per second for output video", default=16)
    guidance_scale: float = Field(
        description="Guidance scale for I2V generation", default=5.0
    )


class AppOutput(BaseAppOutput):
    video_output: File = Field(description="The generated video File")


class App(BaseApp):
    async def setup(self, metadata):
        """Initialize both T2V and I2V pipelines."""
        # Initialize T2V pipeline
        self.device = "cuda"
        self.i2v_model_id = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
        self.image_encoder = CLIPVisionModel.from_pretrained(
            self.i2v_model_id, subfolder="image_encoder", torch_dtype=torch.float32
        )
        self.vae = AutoencoderKLWan.from_pretrained(
            self.i2v_model_id, subfolder="vae", torch_dtype=torch.float32
        )
        self.i2v_pipe = WanImageToVideoPipeline.from_pretrained(
            self.i2v_model_id,
            vae=self.vae,
            image_encoder=self.image_encoder,
            torch_dtype=torch.bfloat16,
        )
        
        self.i2v_pipe.to(self.device)
        
        self.loaded_lora = ""

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Run either T2V or I2V based on whether an image is provided."""
        output_path = "/tmp/output_video.mp4"
        
        # Assert that exactly one of lora_path, t2v_effect, or i2v_effect is provided
        # options = [bool(input_data.lora_path), bool(input_data.t2v_effect), bool(input_data.i2v_effect)]
        # if sum(options) != 1:
        #     raise ValueError("Must provide exactly one of: lora_path, t2v_effect, or i2v_effect")

        # if input_data.lora_path:
        #     self.i2v_pipe.load_lora_weights(input_data.lora_path)
        
        if input_data.i2v_effect:
            # Download LoRA weights from HF Hub        
            repo_id = i2v_effects[input_data.i2v_effect]["repo"]
            folder = i2v_effects[input_data.i2v_effect]["folder"]
            filename = i2v_effects[input_data.i2v_effect]["file"]
            
            lora_path = hf_hub_download(
                repo_id=repo_id,
                filename=f"{folder}/{filename}",
            )
            
            if self.loaded_lora != lora_path and self.loaded_lora != "":
                self.i2v_pipe.unload_lora_weights()
                
            self.i2v_pipe.load_lora_weights(lora_path)
            self.loaded_lora = lora_path
        else:
            self.i2v_pipe.unload_lora_weights()
            self.loaded_lora = ""

        # Load and preprocess image
        image = load_image(input_data.image.path)
        max_area = 480 * 832
        aspect_ratio = image.height / image.width
        mod_value = (
            self.i2v_pipe.vae_scale_factor_spatial
            * self.i2v_pipe.transformer.config.patch_size[1]
        )
        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        image = image.resize((width, height))

        output = self.i2v_pipe(
            image=image,
            prompt=input_data.prompt,
            negative_prompt=input_data.negative_prompt,
            height=height,
            width=width,
            num_frames=input_data.num_frames,
            guidance_scale=input_data.guidance_scale,
        ).frames[0]

       
        export_to_video(output, output_path, fps=input_data.fps)
        return AppOutput(video_output=File(path=output_path))


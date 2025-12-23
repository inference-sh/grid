from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from typing import Optional
import torch
from diffusers import WanPipeline
from diffusers.utils import export_to_video
from huggingface_hub import hf_hub_download
import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

t2v_effects = {
    "Abandoned-Places": {
        "repo": "inference-sh/remade-ai-wan2.1-t2v-effects",
        "folder": "Abandoned-Places",
        "file": "abandoned_50_epochs.safetensors",
    },
    "Animal-Documentary": {
        "repo": "inference-sh/remade-ai-wan2.1-t2v-effects",
        "folder": "Animal-Documentary",
        "file": "animal_doc_5_epochs.safetensors",
    },
    "Boxing": {
        "repo": "inference-sh/remade-ai-wan2.1-t2v-effects",
        "folder": "Boxing",
        "file": "boxing_10_epochs.safetensors",
    },
    "Cats": {
        "repo": "inference-sh/remade-ai-wan2.1-t2v-effects",
        "folder": "Cats",
        "file": "cats_10_epochs.safetensors",
    },
    "Cyberpunk": {
        "repo": "inference-sh/remade-ai-wan2.1-t2v-effects",
        "folder": "Cyberpunk",
        "file": "cyberpunk_20_epochs.safetensors",
    },
    "Dogs": {
        "repo": "inference-sh/remade-ai-wan2.1-t2v-effects",
        "folder": "Dogs",
        "file": "dogs_5_epochs.safetensors",
    },
    "Doom-FPS": {
        "repo": "inference-sh/remade-ai-wan2.1-t2v-effects",
        "folder": "Doom-FPS",
        "file": "doom_8_epochs.safetensors",
    },
    "Eye-Close-Up": {
        "repo": "inference-sh/remade-ai-wan2.1-t2v-effects",
        "folder": "Eye-Close-Up",
        "file": "eye_10_epochs.safetensors",
    },
    "Fantasy-Landscapes": {
        "repo": "inference-sh/remade-ai-wan2.1-t2v-effects",
        "folder": "Fantasy-Landscapes",
        "file": "fantasy_50_epochs.safetensors",
    },
    "Film-Noir": {
        "repo": "inference-sh/remade-ai-wan2.1-t2v-effects",
        "folder": "Film-Noir",
        "file": "film_noir_10_epochs.safetensors",
    },
    "Fire": {
        "repo": "inference-sh/remade-ai-wan2.1-t2v-effects",
        "folder": "Fire",
        "file": "fire_12_epochs.safetensors",
    },
    "Lego": {
        "repo": "inference-sh/remade-ai-wan2.1-t2v-effects",
        "folder": "Lego",
        "file": "lego_35_epochs.safetensors",
    },
    "POV-Driving": {
        "repo": "inference-sh/remade-ai-wan2.1-t2v-effects",
        "folder": "POV-Driving",
        "file": "pov_driving_5_epochs.safetensors",
    },
    "Pixar": {
        "repo": "inference-sh/remade-ai-wan2.1-t2v-effects",
        "folder": "Pixar",
        "file": "pixar_10_epochs.safetensors",
    },
    "Tiny-Planet-Fisheye": {
        "repo": "inference-sh/remade-ai-wan2.1-t2v-effects",
        "folder": "Tiny-Planet-Fisheye",
        "file": "fisheye_15_epochs.safetensors",
    },
    "Tornado": {
        "repo": "inference-sh/remade-ai-wan2.1-t2v-effects",
        "folder": "Tornado",
        "file": "tornado_10_epochs.safetensors",
    },
    "Tsunami": {
        "repo": "inference-sh/remade-ai-wan2.1-t2v-effects",
        "folder": "Tsunami",
        "file": "tsunami_4_epochs.safetensors",
    },
    "Ultra-Wide": {
        "repo": "inference-sh/remade-ai-wan2.1-t2v-effects",
        "folder": "Ultra-Wide",
        "file": "ultra_wide_5_epochs.safetensors",
    },
    "Vintage-VHS": {
        "repo": "inference-sh/remade-ai-wan2.1-t2v-effects",
        "folder": "Vintage-VHS",
        "file": "vhs_20_epochs.safetensors",
    },
    "Zoom-Call": {
        "repo": "inference-sh/remade-ai-wan2.1-t2v-effects",
        "folder": "Zoom-Call",
        "file": "zoom_call_10_epochs.safetensors",
    },
}


class AppInput(BaseAppInput):
    prompt: str = Field(description="The input prompt describing the video to generate")
    t2v_effect: str = Field(description="Choose an effect", enum=list(t2v_effects.keys()))
    negative_prompt: str = Field(
        description="The negative prompt to guide generation", default=""
    )
    num_frames: int = Field(description="Number of frames to generate", default=33)
    fps: int = Field(description="Frames per second for output video", default=16)


class AppOutput(BaseAppOutput):
    video_output: File = Field(description="The generated video File")


class App(BaseApp):
    async def setup(self, metadata):
        """Initialize T2V pipeline."""
        self.device = "cuda"
        self.t2v_model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
        self.t2v_pipe = WanPipeline.from_pretrained(
            self.t2v_model_id, torch_dtype=torch.bfloat16
        )
        self.t2v_pipe.to(self.device)

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Run T2V generation."""
        output_path = "/tmp/output_video.mp4"
        
        # Download LoRA weights from HF Hub        
        repo_id = t2v_effects[input_data.t2v_effect]["repo"]
        folder = t2v_effects[input_data.t2v_effect]["folder"]
        filename = t2v_effects[input_data.t2v_effect]["file"]
        
        lora_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{folder}/{filename}",
        )
        self.t2v_pipe.load_lora_weights(lora_path)

        output = self.t2v_pipe(
            prompt=input_data.prompt,
            negative_prompt=input_data.negative_prompt,
            num_frames=input_data.num_frames,
        ).frames[0]

        export_to_video(output, output_path, fps=input_data.fps)
        return AppOutput(video_output=File(path=output_path))


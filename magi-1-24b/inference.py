import os
import sys
sys.path.append(os.path.dirname(__file__))  # Add just the current directory

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field

from magi.pipeline import MagiPipeline
from typing import Optional
from enum import Enum
import json
from huggingface_hub import snapshot_download

class Mode(str, Enum):
    T2V = "t2v"
    I2V = "i2v"
    V2V = "v2v"

class AppInput(BaseAppInput):
    mode: Mode = Field(description="The mode of the pipeline")
    prompt: str = Field(description="The prompt for the pipeline. Used for all modes.")
    image_path: Optional[str] = Field(None, description="For i2v mode, the source image path")
    prefix_video_path: Optional[str] = Field(None, description="For v2v mode, the source video path")

class AppOutput(BaseAppOutput):
    video: File

class App(BaseApp):
    async def setup(self):
        """Initialize the MAGI-1 pipeline."""
        config_file = os.path.join(os.path.dirname(__file__), "example/24B/24B_config.json")
        config_json = json.load(open(config_file))

        # 1. Download MAGI base weights
        magi_weights_path = snapshot_download(
            repo_id="sand-ai/MAGI-1",
            allow_patterns=["ckpt/magi/24B_base/inference_weight/*.safetensors", "ckpt/magi/24B_base/inference_weight/*.json"],
        )
        config_json["runtime_config"]["load"] = os.path.join(magi_weights_path, "ckpt/magi/24B_base/inference_weight")

        # 2. Download T5 model files
        t5_path = snapshot_download(
            repo_id="sand-ai/MAGI-1",
            allow_patterns=[
                "ckpt/t5/t5-v1_1-xxl/*.bin",
                "ckpt/t5/t5-v1_1-xxl/*.json",
                "ckpt/t5/t5-v1_1-xxl/spiece.model"
            ],
        )
        config_json["runtime_config"]["t5_pretrained"] = os.path.join(t5_path, "ckpt/t5/t5-v1_1-xxl")

        # 3. Download VAE model files
        vae_path = snapshot_download(
            repo_id="sand-ai/MAGI-1",
            allow_patterns=[
                "ckpt/vae/*.safetensors",
                "ckpt/vae/config.json"
            ],
        )
        config_json["runtime_config"]["vae_pretrained"] = os.path.join(vae_path, "ckpt/vae")
        
        # Write updated config back to file
        with open(config_file, "w") as f:
            json.dump(config_json, f, indent=4)
        self.pipeline = MagiPipeline(config_file)

    async def run(self, input_data: AppInput) -> AppOutput:
        """Run the appropriate MAGI pipeline based on input mode."""
        output_path = "/tmp/output.mp4"
        
        if input_data.mode == Mode.T2V:
            self.pipeline.run_text_to_video(
                prompt=input_data.prompt,
                output_path=output_path
            )
        elif input_data.mode == Mode.I2V:
            if not input_data.image_path:
                raise ValueError("image_path is required for i2v mode")
            self.pipeline.run_image_to_video(
                prompt=input_data.prompt,
                image_path=input_data.image_path,
                output_path=output_path
            )
        elif input_data.mode == Mode.V2V:
            if not input_data.prefix_video_path:
                raise ValueError("prefix_video_path is required for v2v mode")
            self.pipeline.run_video_to_video(
                prompt=input_data.prompt,
                prefix_video_path=input_data.prefix_video_path,
                output_path=output_path
            )
        
        return AppOutput(video=File(path=output_path))

    async def unload(self):
        """Clean up resources."""
        # Add any cleanup code if needed
        pass
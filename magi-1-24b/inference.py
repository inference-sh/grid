import os
import sys

sys.path.append(os.path.dirname(__file__))  # Add just the current directory

import gc
import json
from enum import Enum
from typing import Optional

import torch
from huggingface_hub import snapshot_download
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from magi.pipeline import MagiPipeline
from pydantic import Field


class Mode(str, Enum):
    T2V = "t2v"
    I2V = "i2v"
    V2V = "v2v"

VIDEO_SIZES = ["480p", "720p"]

class AppInput(BaseAppInput):
    mode: Mode = Field(description="The mode of the pipeline")
    prompt: str = Field(description="The prompt for the pipeline. Used for all modes.")
    image: Optional[File] = Field(None, description="For i2v mode, the source image path")
    prefix_video: Optional[File] = Field(None, description="For v2v mode, the source video path")
    seed: int = Field(-1, description="Random seed. -1 means random.")
    num_frames: int = Field(96, description="Number of frames in the output video.")
    num_steps: int = Field(8, description="Number of inference steps.")
    window_size: int = Field(4, description="Window size for inference.")
    fps: int = Field(24, description="Frames per second.")
    chunk_width: int = Field(6, description="Chunk width for inference.")
    video_size: str = Field("720p", description="Video size: 480p or 720p.", enum=VIDEO_SIZES)

class AppOutput(BaseAppOutput):
    video: File

class App(BaseApp):
    def __init__(self):
        super().__init__()
        self._last_runtime_config = None
        self.pipeline = None
        self._weights_paths = {}

    async def setup(self):
        """Download weights and store paths, but do not write config file."""
        config_file = os.path.join(os.path.dirname(__file__), "example/24B/24B_config.json")
        with open(config_file, "r") as f:
            config_json = json.load(f)

        # 1. Download MAGI base weights
        magi_weights_path = snapshot_download(
            repo_id="sand-ai/MAGI-1",
            allow_patterns=["ckpt/magi/24B_base/inference_weight/*.safetensors", "ckpt/magi/24B_base/inference_weight/*.json"],
        )
        self._weights_paths["load"] = os.path.join(magi_weights_path, "ckpt/magi/24B_base/")

        # 2. Download T5 model files
        t5_path = snapshot_download(
            repo_id="sand-ai/MAGI-1",
            allow_patterns=[
                "ckpt/t5/t5-v1_1-xxl/*.bin",
                "ckpt/t5/t5-v1_1-xxl/*.json",
                "ckpt/t5/t5-v1_1-xxl/spiece.model"
            ],
        )
        self._weights_paths["t5_pretrained"] = os.path.join(t5_path, "ckpt/t5/")

        # 3. Download VAE model files
        vae_path = snapshot_download(
            repo_id="sand-ai/MAGI-1",
            allow_patterns=[
                "ckpt/vae/*.safetensors",
                "ckpt/vae/config.json"
            ],
        )
        self._weights_paths["vae_pretrained"] = os.path.join(vae_path, "ckpt/vae/")

    async def run(self, input_data: AppInput) -> AppOutput:
        import random
        output_path = "/tmp/output.mp4"
        config_file = os.path.join(os.path.dirname(__file__), "example/24B/24B_config.json")
        with open(config_file, "r") as f:
            config_json = json.load(f)

        # Always update weights paths in config
        config_json["runtime_config"]["load"] = self._weights_paths["load"]
        config_json["runtime_config"]["t5_pretrained"] = self._weights_paths["t5_pretrained"]
        config_json["runtime_config"]["vae_pretrained"] = self._weights_paths["vae_pretrained"]

        # Build the relevant config dict from input_data
        if input_data.seed == -1:
            seed = random.randint(0, 2**31 - 1)
        else:
            seed = input_data.seed

        if input_data.video_size == "480p":
            video_size_h, video_size_w = 480, 854
        else:
            video_size_h, video_size_w = 720, 1280

        rc = config_json["runtime_config"]
        rc["seed"] = seed
        rc["num_frames"] = input_data.num_frames
        rc["num_steps"] = input_data.num_steps
        rc["window_size"] = input_data.window_size
        rc["fps"] = input_data.fps
        rc["chunk_width"] = input_data.chunk_width
        rc["video_size_h"] = video_size_h
        rc["video_size_w"] = video_size_w

        # Write config ONCE per run
        with open(config_file, "w") as f:
            json.dump(config_json, f, indent=4)

        # Re-instantiate pipeline if config changed
        if self._last_runtime_config != rc or self.pipeline is None:
            if self.pipeline is not None:
                del self.pipeline
                torch.cuda.empty_cache()
                gc.collect()
            self.pipeline = MagiPipeline(config_file)
            self._last_runtime_config = rc

        # Now run the pipeline as before
        if input_data.mode == Mode.T2V:
            self.pipeline.run_text_to_video(
                prompt=input_data.prompt,
                output_path=output_path
            )
        elif input_data.mode == Mode.I2V:
            if not input_data.image:
                raise ValueError("image is required for i2v mode")
            self.pipeline.run_image_to_video(
                prompt=input_data.prompt,
                image_path=input_data.image.path,
                output_path=output_path
            )
        elif input_data.mode == Mode.V2V:
            if not input_data.prefix_video:
                raise ValueError("prefix_video is required for v2v mode")
            self.pipeline.run_video_to_video(
                prompt=input_data.prompt,
                prefix_video_path=input_data.prefix_video.path,
                output_path=output_path
            )
        
        return AppOutput(video=File(path=output_path))

    async def unload(self):
        """Clean up resources."""
        # Add any cleanup code if needed
        pass
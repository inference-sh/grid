import os
import sys

sys.path.append(os.path.dirname(__file__))

import gc
import json
import pickle
import random
from enum import Enum
from typing import Optional

import torch
import torch.distributed as dist
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
    image: Optional[File] = Field(
        None, description="For i2v mode, the source image path"
    )
    prefix_video: Optional[File] = Field(
        None, description="For v2v mode, the source video path"
    )
    seed: int = Field(-1, description="Random seed. -1 means random.")
    num_frames: int = Field(96, description="Number of frames in the output video.")
    num_steps: int = Field(8, description="Number of inference steps.")
    window_size: int = Field(4, description="Window size for inference.")
    fps: int = Field(24, description="Frames per second.")
    chunk_width: int = Field(6, description="Chunk width for inference.")
    width: int = Field(1280, description="Width of the output video. Must be multiple of 8.")
    height: int = Field(720, description="Height of the output video. Must be multiple of 8.")


class AppOutput(BaseAppOutput):
    video: File


def get_device(local_rank=None):
    backend = torch.distributed.get_backend()
    if backend == "nccl":
        if local_rank is None:
            device = torch.device("cuda")
        else:
            device = torch.device(f"cuda:{local_rank}")
    elif backend == "gloo":
        device = torch.device("cpu")
    else:
        raise RuntimeError
    return device

configs = {
    "default": {
        "config_file": "example/4.5B/4.5B_base_config.json",
        "weights_base_path": "4.5B_base",
        "weights_path": "4.5B_base/inference_weight"
    },
    "24B_base": {
        "config_file": "example/24B/24B_base_config.json",
        "weights_base_path": "24B_base",
        "weights_path": "24B_base/inference_weight"
    },
    "24B_distill": {
        "config_file": "example/24B/24B_distill_config.json",
        "weights_base_path": "24B_distill",
        "weights_path": "24B_distill/inference_weight.distill"
    },
    "24B_distill_quant": {
        "config_file": "example/24B/24B_distill_quant_config.json",
        "weights_base_path": "24B_distill_quant",
        "weights_path": "24B_distill_quant/inference_weight.fp8.distill"
    }
}


def broadcast_config(config_json: Optional[dict]) -> dict:
    if dist.is_available() and dist.is_initialized():
        device = get_device()

        if dist.get_rank() == 0:
            data = pickle.dumps(config_json)
            size = torch.tensor([len(data)], dtype=torch.long, device=device)
        else:
            size = torch.empty(1, dtype=torch.long, device=device)

        dist.broadcast(size, src=0)

        if dist.get_rank() == 0:
            tensor = torch.tensor(list(data), dtype=torch.uint8, device=device)
        else:
            tensor = torch.empty(size.item(), dtype=torch.uint8, device=device)

        dist.broadcast(tensor, src=0)

        return pickle.loads(bytearray(tensor.cpu().tolist()))
    return config_json


class App(BaseApp):
    def __init__(self):
        super().__init__()
        self._last_runtime_config = None
        self.pipeline = None
        self._weights_paths = {}

    async def setup(self, metadata):
        if isinstance(metadata, dict):
            self.variant_config = configs[metadata.get("app_variant", "default")]
        else:
            self.variant_config = configs[metadata.app_variant]

        magi_weights_path = snapshot_download(
            repo_id="sand-ai/MAGI-1",
            allow_patterns=[
                f"ckpt/magi/{self.variant_config['weights_path']}/*.safetensors",
                f"ckpt/magi/{self.variant_config['weights_path']}/*.json",
            ],
        )
        self._weights_paths["load"] = os.path.join(
            magi_weights_path, f"ckpt/magi/{self.variant_config['weights_base_path']}/"
        )

        t5_path = snapshot_download(
            repo_id="sand-ai/MAGI-1",
            allow_patterns=[
                "ckpt/t5/t5-v1_1-xxl/*.bin",
                "ckpt/t5/t5-v1_1-xxl/*.json",
                "ckpt/t5/t5-v1_1-xxl/spiece.model",
            ],
        )
        self._weights_paths["t5_pretrained"] = os.path.join(t5_path, "ckpt/t5/")

        vae_path = snapshot_download(
            repo_id="sand-ai/MAGI-1",
            allow_patterns=["ckpt/vae/*.safetensors", "ckpt/vae/config.json"],
        )
        self._weights_paths["vae_pretrained"] = os.path.join(vae_path, "ckpt/vae/")

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        output_path = "/tmp/output.mp4"
        config_file = os.path.join(
            os.path.dirname(__file__), self.variant_config["config_file"]
        )
        is_primary = (
            dist.get_rank() == 0
            if dist.is_available() and dist.is_initialized()
            else True
        )

        if is_primary:
            with open(config_file, "r") as f:
                config_json = json.load(f)

            config_json["runtime_config"]["load"] = self._weights_paths["load"]
            config_json["runtime_config"]["t5_pretrained"] = self._weights_paths[
                "t5_pretrained"
            ]
            config_json["runtime_config"]["vae_pretrained"] = self._weights_paths[
                "vae_pretrained"
            ]

            seed = (
                input_data.seed
                if input_data.seed != -1
                else random.randint(0, 2**31 - 1)
            )
            h, w = (input_data.height, input_data.width)

            rc = config_json["runtime_config"]
            rc.update(
                {
                    "seed": seed,
                    "num_frames": input_data.num_frames,
                    "num_steps": input_data.num_steps,
                    "window_size": input_data.window_size,
                    "fps": input_data.fps,
                    "chunk_width": input_data.chunk_width,
                    "video_size_h": h,
                    "video_size_w": w,
                }
            )

            with open(config_file, "w") as f:
                json.dump(config_json, f, indent=4)
        else:
            config_json = None

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            config_json = broadcast_config(config_json)

        rc = config_json["runtime_config"]
        if self._last_runtime_config != rc or self.pipeline is None:
            if self.pipeline is not None:
                del self.pipeline
                torch.cuda.empty_cache()
                gc.collect()
            self.pipeline = MagiPipeline(config_file)
            self._last_runtime_config = rc

        if input_data.mode == Mode.T2V:
            self.pipeline.run_text_to_video(
                prompt=input_data.prompt, output_path=output_path
            )
        elif input_data.mode == Mode.I2V:
            if not input_data.image:
                raise ValueError("image is required for i2v mode")
            self.pipeline.run_image_to_video(
                prompt=input_data.prompt,
                image_path=input_data.image.path,
                output_path=output_path,
            )
        elif input_data.mode == Mode.V2V:
            if not input_data.prefix_video:
                raise ValueError("prefix_video is required for v2v mode")
            self.pipeline.run_video_to_video(
                prompt=input_data.prompt,
                prefix_video_path=input_data.prefix_video.path,
                output_path=output_path,
            )

        return AppOutput(video=File(path=output_path))

    async def unload(self):
        pass

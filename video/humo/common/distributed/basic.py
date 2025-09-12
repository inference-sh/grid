# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Codes adapted from [SeedVR]
# https://github.com/ByteDance-Seed/SeedVR/tree/main/common/distributed

"""
Distributed basic functions.
"""

import os
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.fsdp._common_utils import _is_fsdp_flattened


def get_global_rank() -> int:
    """
    Get the global rank, the global index of the GPU.
    """
    return int(os.environ.get("RANK", "0"))


def get_local_rank() -> int:
    """
    Get the local rank, the local index of the GPU.
    """
    return int(os.environ.get("LOCAL_RANK", "0"))


def get_world_size() -> int:
    """
    Get the world size, the total amount of GPUs.
    """
    return int(os.environ.get("WORLD_SIZE", "1"))


def get_device() -> torch.device:
    """
    Get current rank device.
    """
    return torch.device("cuda", get_local_rank())


def barrier_if_distributed(*args, **kwargs):
    """
    Synchronizes all processes if under distributed context.
    """
    if dist.is_initialized():
        return dist.barrier(*args, **kwargs)


def init_torch(cudnn_benchmark=True):
    """
    Common PyTorch initialization configuration.
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.cuda.set_device(get_local_rank())
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            rank=get_global_rank(),
            world_size=get_world_size(),
        )


def convert_to_ddp(module: torch.nn.Module, **kwargs) -> DistributedDataParallel:
    return DistributedDataParallel(
        module=module,
        device_ids=[get_local_rank()],
        output_device=get_local_rank(),
        **kwargs,
    )


def meta_param_init_fn(module: nn.Module) -> None:
    """
    Used for model inited onto meta device.
    Init meta param/buffer with empty tensor.
    We don't care numerical correctness in this func.
    FSDP will sync param/buffer state from rank0 to the other ranks.
    """

    with torch.no_grad():
        for submodule in module.modules():
            for param_name, param in submodule.named_parameters(recurse=False):
                if not _is_fsdp_flattened(param) and param.is_meta:
                    materialized_param = nn.Parameter(torch.empty_like(param, device="cpu"))
                    setattr(submodule, param_name, materialized_param)
            for buffer_name, buffer in submodule.named_buffers(recurse=False):
                if not _is_fsdp_flattened(buffer) and buffer.is_meta:
                    materialized_param = torch.empty_like(buffer, device="cpu")
                    setattr(submodule, buffer_name, materialized_param)
            torch.cuda.empty_cache()


def meta_non_persistent_buffer_init_fn(module: nn.Module) -> nn.Module:
    """
    Materialize meta device buffers that are not persistent in state_dict.
    Handles special cases like RotaryEmbedding.freqs.
    """
    with torch.no_grad():
        for submodule in module.modules():
            if hasattr(submodule, "freqs"):
                freqs = getattr(submodule, "freqs")
                if isinstance(freqs, torch.Tensor) and freqs.is_meta:
                    dim = submodule.dim
                    def rope_params(max_seq_len, dim, theta=10000):
                        assert dim % 2 == 0
                        freqs = torch.outer(
                            torch.arange(max_seq_len),
                            1.0 / torch.pow(theta,
                                torch.arange(0, dim, 2).to(torch.float32).div(dim)))
                        freqs = torch.polar(torch.ones_like(freqs), freqs)
                        return freqs
                    
                    dim = 5120  # 1536 
                    num_heads = 40  # 12
                    # dim = 1536 
                    # num_heads = 12
                    d = dim // num_heads
                    freqs_tensor = torch.cat([
                        rope_params(1024, d - 4 * (d // 6)),
                        rope_params(1024, 2 * (d // 6)),
                        rope_params(1024, 2 * (d // 6))
                    ], dim=1).to(dtype=torch.cfloat, device="cpu")
                    
                    setattr(submodule, "freqs", freqs_tensor)
                    print(f"Successfully materialized freqs for {submodule.__class__.__name__}")

    assert not any(b.is_meta for n, b in module.named_buffers())
    return module

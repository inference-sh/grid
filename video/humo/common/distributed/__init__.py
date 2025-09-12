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
Distributed package.
"""

from .basic import (
    barrier_if_distributed,
    convert_to_ddp,
    get_device,
    get_global_rank,
    get_local_rank,
    get_world_size,
    init_torch,
    meta_param_init_fn,
    meta_non_persistent_buffer_init_fn
)

__all__ = [
    "barrier_if_distributed",
    "convert_to_ddp",
    "get_device",
    "get_global_rank",
    "get_local_rank",
    "get_world_size",
    "init_torch",
    "meta_param_init_fn",
    "meta_non_persistent_buffer_init_fn",
]

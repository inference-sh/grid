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
Advanced distributed functions for sequence parallel.
"""

import torch
from typing import Any, List, Optional, Tuple, Union
import torch.distributed as dist
from torch import Tensor

from .basic import get_global_rank, get_world_size


_DATA_PARALLEL_GROUP = None
_SEQUENCE_PARALLEL_GROUP = None
_SEQUENCE_PARALLEL_CPU_GROUP = None


_CFG_PARALLEL_GROUP = None
_CFG_PARALLEL_CPU_GROUP = None

def get_data_parallel_group() -> Optional[dist.ProcessGroup]:
    """
    Get data parallel process group.
    """
    return _DATA_PARALLEL_GROUP


def get_sequence_parallel_group() -> Optional[dist.ProcessGroup]:
    """
    Get sequence parallel process group.
    """
    return _SEQUENCE_PARALLEL_GROUP


def get_sequence_parallel_cpu_group() -> Optional[dist.ProcessGroup]:
    """
    Get sequence parallel CPU process group.
    """
    return _SEQUENCE_PARALLEL_CPU_GROUP


def get_data_parallel_rank() -> int:
    """
    Get data parallel rank.
    """
    group = get_data_parallel_group()
    return dist.get_rank(group) if group else get_global_rank()


def get_data_parallel_world_size() -> int:
    """
    Get data parallel world size.
    """
    group = get_data_parallel_group()
    return dist.get_world_size(group) if group else get_world_size()


def get_sequence_parallel_rank() -> int:
    """
    Get sequence parallel rank.
    """
    group = get_sequence_parallel_group()
    return dist.get_rank(group) if group else 0


def get_sequence_parallel_world_size() -> int:
    """
    Get sequence parallel world size.
    """
    group = get_sequence_parallel_group()
    return dist.get_world_size(group) if group else 1


def init_unified_parallel(unified_parallel_size):
    global _SEQUENCE_PARALLEL_GROUP
    global _SEQUENCE_PARALLEL_CPU_GROUP

    if unified_parallel_size == 1:
        return

    assert dist.is_initialized()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    assert world_size % unified_parallel_size == 0
    data_parallel_size = world_size // unified_parallel_size

    for i in range(data_parallel_size):
        # build unified parallel group
        start_rank = i * unified_parallel_size
        end_rank = start_rank + unified_parallel_size
        unified_parallel_ranks = range(start_rank, end_rank)
        unified_parallel_group = dist.new_group(unified_parallel_ranks)
        unified_parallel_cpu_group = dist.new_group(unified_parallel_ranks, backend="gloo")
        if rank in unified_parallel_ranks:
            _SEQUENCE_PARALLEL_GROUP = unified_parallel_group
            _SEQUENCE_PARALLEL_CPU_GROUP = unified_parallel_cpu_group


def get_unified_parallel_group():
    global _SEQUENCE_PARALLEL_GROUP
    return _SEQUENCE_PARALLEL_GROUP


def get_unified_parallel_cpu_group():
    global _SEQUENCE_PARALLEL_CPU_GROUP
    return _SEQUENCE_PARALLEL_CPU_GROUP


def get_unified_parallel_rank():
    group = get_unified_parallel_group()
    return dist.get_rank(group) if group else 0


def get_unified_parallel_world_size():
    group = get_unified_parallel_group()
    return dist.get_world_size(group) if group else 1


def is_unified_parallel_initialized():
    group = get_unified_parallel_group()
    return group is not None


def pad_tensor(x: Tensor, dim: int, padding_size: int):
    shape = list(x.shape)
    shape[dim] = padding_size
    pad = torch.zeros(shape, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=dim)


class Slice(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, local_input: Tensor, dim: int, scale_grad: bool) -> Tensor:
        ctx.group = group
        ctx.rank = dist.get_rank(group)
        seq_world_size = dist.get_world_size(group)
        ctx.seq_world_size = seq_world_size
        ctx.dim = dim
        ctx.scale_grad = scale_grad
        dim_size = local_input.shape[dim]
        if not ctx.group:
            return local_input
        return local_input.split(dim_size // seq_world_size, dim=dim)[ctx.rank].contiguous()

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[None, Tensor, None]:
        if not ctx.group:
            return None, grad_output, None, None
        dim_size = list(grad_output.size())
        split_size = dim_size[0]
        dim_size[0] = dim_size[0] * ctx.seq_world_size
        output = torch.empty(dim_size, dtype=grad_output.dtype, device=torch.cuda.current_device())
        dist.all_gather_into_tensor(output, grad_output, group=ctx.group)
        if ctx.scale_grad:
            output = output / ctx.seq_world_size
        return (None, torch.cat(output.split(split_size), dim=ctx.dim), None, None)


def gather_outputs(
    x: Tensor,
    gather_dim: int,
    padding_dim: Optional[int] = None,
    unpad_dim_size: Optional[int] = None,
    scale_grad=True,
):
    """
    A func to gather the outputs for the model result in sequence parallel
    """
    group = get_unified_parallel_group()
    if not group:
        return x
    x = Gather.apply(group, x, gather_dim, scale_grad)
    if padding_dim is not None:
        x = unpadding_tensor_for_seqeunce_parallel(x, padding_dim, unpad_dim_size)
    return x


def unpadding_tensor_for_seqeunce_parallel(x: Tensor, dim: int, unpadded_dim_size: int):
    """
    A func to remove the padding part of the tensor based on its original shape
    """
    group = get_unified_parallel_group()
    if group is None:
        return x
    sp_world = get_unified_parallel_world_size()
    if unpadded_dim_size % sp_world == 0:
        return x
    padding_size = sp_world - (unpadded_dim_size % sp_world)
    assert (padding_size + unpadded_dim_size) % sp_world == 0
    return unpad_tensor(x, dim=dim, padding_size=padding_size)


def gather_seq_scatter_heads_qkv(
    qkv_tensor: Tensor,
    seq_dim: int,
    unpadded_dim_size: Optional[int] = None,
    restore_shape: bool = True,
    async_op: bool = False,
):
    """
    A func to sync splited qkv tensor
    qkv_tensor: the tensor we want to do alltoall with. The last dim must
        be the projection_idx, which we will split into 3 part. After
        spliting, the gather idx will be projecttion_idx + 1
    seq_dim: gather_dim for all2all comm
    restore_shape: if True, output will has the same shape length as input
    """
    group = get_unified_parallel_group()
    if not group:
        return qkv_tensor
    world = get_unified_parallel_world_size()
    orig_shape = qkv_tensor.shape
    scatter_dim = qkv_tensor.dim()
    bef_all2all_shape = list(orig_shape)
    qkv_proj_dim = bef_all2all_shape[-1]
    bef_all2all_shape = bef_all2all_shape[:-1] + [3, qkv_proj_dim // 3]
    qkv_tensor = qkv_tensor.view(bef_all2all_shape)
    if async_op:
        return SeqAllToAll.apply(group, qkv_tensor, scatter_dim, seq_dim, async_op)
    else:
        qkv_tensor = SeqAllToAll.apply(group, qkv_tensor, scatter_dim, seq_dim, async_op)

        if restore_shape:
            out_shape = list(orig_shape)
            out_shape[seq_dim] *= world
            out_shape[-1] = qkv_proj_dim // world
            qkv_tensor = qkv_tensor.view(out_shape)

        # remove padding
        if unpadded_dim_size and unpadded_dim_size % world != 0:
            padding_size = qkv_tensor.size(seq_dim) - unpadded_dim_size
            qkv_tensor = unpad_tensor(qkv_tensor, seq_dim, padding_size)

        return qkv_tensor


def gather_seq_scatter_double_head(
    qkv_tensor: Tensor,
    seq_dim: int,
    unpadded_dim_size: Optional[int] = None,
    restore_shape: bool = True,
    async_op: bool = False,
):
    """
    A func to sync splited qkv tensor
    qkv_tensor: the tensor we want to do alltoall with. The last dim must
        be the projection_idx, which we will split into 3 part. After
        spliting, the gather idx will be projecttion_idx + 1
    seq_dim: gather_dim for all2all comm
    restore_shape: if True, output will has the same shape length as input
    """
    qkv1_shape = qkv_tensor.shape
    group = get_unified_parallel_group()
    if not group:
        return qkv_tensor
    world = get_unified_parallel_world_size()
    orig_shape = qkv_tensor.shape
    scatter_dim = qkv_tensor.dim()
    bef_all2all_shape = list(orig_shape)
    qkv_proj_dim = bef_all2all_shape[-1]
    bef_all2all_shape = bef_all2all_shape[:-1] + [2, qkv_proj_dim // 2]
    qkv_tensor = qkv_tensor.view(bef_all2all_shape)
    qkv2_shape = qkv_tensor.shape
    if async_op:
        return SeqAllToAll.apply(group, qkv_tensor, scatter_dim, seq_dim, async_op)
    else:
        qkv_tensor = SeqAllToAll.apply(group, qkv_tensor, scatter_dim, seq_dim, async_op)
        qkv3_shape = qkv_tensor.shape

        if restore_shape:
            out_shape = list(orig_shape)
            out_shape[seq_dim] *= world
            out_shape[-1] = qkv_proj_dim // world
            qkv_tensor = qkv_tensor.view(out_shape)
            qkv4_shape = qkv_tensor.shape

        # remove padding
        if unpadded_dim_size and unpadded_dim_size % world != 0:
            padding_size = qkv_tensor.size(seq_dim) - unpadded_dim_size
            qkv_tensor = unpad_tensor(qkv_tensor, seq_dim, padding_size)
            qkv5_shape = qkv_tensor.shape
        
        return qkv_tensor


class SeqAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        local_input: Tensor,
        scatter_dim: int,
        gather_dim: int,
        async_op: bool,
    ) -> Tensor:
        ctx.group = group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.async_op = async_op
        return all_to_all_tensor(local_input, scatter_dim, gather_dim, group, async_op)

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor, None, None]:
        if ctx.async_op:
            input_t = torch.cat(grad_output[1:], dim=ctx.gather_dim).contiguous()
        else:
            input_t = grad_output[0]
        return (
            None,
            all_to_all_tensor(input_t, ctx.gather_dim, ctx.scatter_dim, ctx.group, False),
            None,
            None,
            None,
            None,
        )


def all_to_all_tensor(
    x: Tensor,
    scatter_dim: int,
    gather_dim: int,
    group: dist.ProcessGroup,
    async_op: bool = False,
):
    if scatter_dim <= 1 and gather_dim <= 1:
        return _all_to_all_single(x, scatter_dim, gather_dim, group, async_op)
    else:
        return _all_to_all(x, scatter_dim, gather_dim, group, async_op)  # 走这里


def _all_to_all(
    local_input: Tensor,
    scatter_dim: int,
    gather_dim: int,
    group: dist.ProcessGroup,
    async_op: bool = False,
):
    seq_world_size = dist.get_world_size(group)
    input_list = [t.contiguous() for t in torch.tensor_split(local_input, seq_world_size, scatter_dim)]
    output_list = [torch.empty_like(input_list[0]) for _ in range(seq_world_size)]
    comm = dist.all_to_all(output_list, input_list, group=group, async_op=async_op)
    if async_op:

        def wait():
            comm.wait()
            return torch.cat(output_list, dim=gather_dim).contiguous()

        return wait
    return torch.cat(output_list, dim=gather_dim).contiguous()


def _all_to_all_single(x: Tensor, scatter_dim: int, gather_dim: int, group: dist.ProcessGroup, async_op: bool = False):
    """
    A function to do all-to-all on the first two dim
    """
    sp_world_size = dist.get_world_size(group)
    assert scatter_dim <= 1, "scatter_dim must be 0 or 1 when using all_to_all_single!"
    assert gather_dim <= 1, "gather_dim must be 0 or 1 when using all_to_all_single!"
    if scatter_dim != 0:
        gather_dim_bef = x.shape[gather_dim]
        scatter_dim_bef = x.shape[scatter_dim]
        x = (
            x.reshape([gather_dim_bef, sp_world_size, scatter_dim_bef // sp_world_size] + list(x.shape[2:]))
            .transpose(0, 1)
            .reshape([gather_dim_bef * sp_world_size, scatter_dim_bef // sp_world_size] + list(x.shape[2:]))
            .contiguous()
        )

    output = torch.empty_like(x)
    comm = dist.all_to_all_single(output, x.contiguous(), group=group, async_op=async_op)

    if async_op:

        def wait():
            comm.wait()
            if scatter_dim == 0:
                return torch.cat(output.split(x.size(0) // sp_world_size), dim=gather_dim)
            else:
                return output

        return wait

    if scatter_dim == 0:
        output = torch.cat(output.split(x.size(0) // sp_world_size), dim=gather_dim)
    return output


def gather_heads_scatter_seq(x: Tensor, head_dim: int, seq_dim: int) -> Tensor:
    """
    A func to sync attention result with alltoall in sequence parallel
    """
    group = get_unified_parallel_group()
    if not group:
        return x
    dim_size = x.size(seq_dim)
    sp_world = get_unified_parallel_world_size()
    if dim_size % sp_world != 0:
        padding_size = sp_world - (dim_size % sp_world)
        x = pad_tensor(x, seq_dim, padding_size)
    return SeqAllToAll.apply(group, x, seq_dim, head_dim, False)


def unpad_tensor(x: Tensor, dim: int, padding_size: int):
    slc = [slice(None)] * len(x.shape)
    slc[dim] = slice(0, -padding_size)
    return x[slc]


class Gather(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        local_input: Tensor,
        dim: int,
        grad_scale: Optional[bool] = False,
    ) -> Tensor:
        ctx.group = group
        ctx.rank = dist.get_rank(group)
        ctx.dim = dim
        ctx.grad_scale = grad_scale
        seq_world_size = dist.get_world_size(group)
        ctx.seq_world_size = seq_world_size
        dim_size = list(local_input.size())
        split_size = dim_size[0]
        ctx.part_size = dim_size[dim]
        dim_size[0] = dim_size[0] * seq_world_size
        output = torch.empty(dim_size, dtype=local_input.dtype, device=torch.cuda.current_device())
        dist.all_gather_into_tensor(output, local_input.contiguous(), group=ctx.group)
        return torch.cat(output.split(split_size), dim=dim)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[None, Tensor]:
        if ctx.grad_scale:
            grad_output = grad_output * ctx.seq_world_size
        return (
            None,
            grad_output.split(ctx.part_size, dim=ctx.dim)[ctx.rank].contiguous(),
            None,
            None,
        )


def slice_tensor(tensor, dim, start, end):
    indices = slice(start, end)
    return tensor[(slice(None),) * dim + (indices,)]


def init_model_shard_cpu_group(sharding_strategy: str, device_mesh: Optional[Tuple] = None):
    """
    Initialize CPU process group of model sharding.
    """
    global _MODEL_SHARD_CPU_GROUP
    assert dist.is_initialized()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if device_mesh is not None:
        num_shards_per_group = device_mesh[1]
    elif "HYBRID" in sharding_strategy:
        num_shards_per_group = min(8, world_size)
    else:
        num_shards_per_group = world_size
    num_groups = world_size // num_shards_per_group
    for i in range(num_groups):
        start_rank = i * num_shards_per_group
        end_rank = (i + 1) * num_shards_per_group
        ranks = range(start_rank, end_rank)
        cpu_group = dist.new_group(ranks, backend="gloo")
        if rank in ranks:
            _MODEL_SHARD_CPU_GROUP = cpu_group

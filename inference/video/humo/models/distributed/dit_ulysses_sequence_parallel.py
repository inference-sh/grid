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

import torch
import torch.cuda.amp as amp
from einops import rearrange
from common.distributed import get_device

from common.distributed.advanced import (
    get_unified_parallel_world_size,
    get_unified_parallel_group,
    pad_tensor,
    Slice,
    gather_outputs,
    gather_seq_scatter_heads_qkv,
    gather_seq_scatter_double_head,
    gather_heads_scatter_seq,
    unpad_tensor
)
from models.wan_modules.attention import flash_attention
from models.wan_modules.model_humo import rope_apply, sinusoidal_embedding_1d


def ulysses_dit_forward(
    self,
    x,
    t,
    context,
    seq_len,
    audio=None,
    y=None
):
    """
    x:              A list of videos each with shape [C, T, H, W].
    t:              [B].
    context:        A list of text embeddings each with shape [L, C].
    """
    if self.model_type == 'i2v':
        # assert clip_fea is not None and y is not None
        assert y is not None
    # params
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    # embeddings
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long, device=device)

    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
        for u in x
    ])

    # time embeddings
    with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).float()).float()
        e0 = self.time_projection(e).unflatten(1, (6, self.dim)).float()
        assert e.dtype == torch.float32 and e0.dtype == torch.float32
        
    # context
    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))

    if self.insert_audio:
        audio = [self.audio_proj(au.unsqueeze(0)).permute(0, 3, 1, 2) for au in audio]

        audio_seq_len = torch.tensor(max([au.shape[2] for au in audio]) * audio[0].shape[3], device=get_device())
        audio = [au.flatten(2).transpose(1, 2) for au in audio] # [1, t*32, 1536]
        audio_seq_lens = torch.tensor([au.size(1) for au in audio], dtype=torch.long, device=device)
        audio = torch.cat([
            torch.cat([au, au.new_zeros(1, audio_seq_len - au.size(1), au.size(2))],
                        dim=1) for au in audio
        ])
    else:
        audio = None
        audio_seq_len = None
        audio_seq_lens = None

    # ulysses support
    sp_world = get_unified_parallel_world_size()
    group = get_unified_parallel_group()
    if seq_len % sp_world:
        padding_size = sp_world - (seq_len % sp_world)
        x = pad_tensor(x, dim=1, padding_size=padding_size)

        if self.insert_audio:
            audio_padding_size = sp_world - (audio_seq_len % sp_world)
            audio = pad_tensor(audio, dim=1, padding_size=audio_padding_size)
        
    x = Slice.apply(group, x, 1, True)

    if self.insert_audio:
        audio = Slice.apply(group, audio, 1, True)

    # arguments
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens,
        audio=audio,
        audio_seq_len=audio_seq_len)
    
    for block in self.blocks:
        x = block(x, **kwargs)

    # head
    x = self.head(x, e)

    # ulysses support
    x = gather_outputs(x, gather_dim=1, padding_dim=1, unpad_dim_size=seq_len, scale_grad=True)
    
    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    return {"latent":[u.float() for u in x], 
            "mask": None}


def ulysses_attn_forward(
    self,
    x,
    seq_lens,
    grid_sizes,
    freqs,
    dtype=torch.bfloat16
):
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    seq_len = seq_lens.max()
    half_dtypes = (torch.float16, torch.bfloat16)

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # query, key, value function
    def qkv_fn(x):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        return q, k, v

    q, k, v = qkv_fn(x)

    # ulysses support
    sp_size = get_unified_parallel_world_size()
    if n % sp_size:
        pad_size = sp_size - (n % sp_size)
        pad_size = pad_size * d
        pad_inner_dim = n * d + pad_size
        q = pad_tensor(q, dim=2, padding_size=pad_size)
        k = pad_tensor(k, dim=2, padding_size=pad_size)
        v = pad_tensor(v, dim=2, padding_size=pad_size)
    else:
        pad_inner_dim = n * d

    qkv = torch.cat([q, k, v], dim=2)
    qkv = gather_seq_scatter_heads_qkv(qkv, seq_dim=1, unpadded_dim_size=seq_len)
    q, k, v = qkv.split(pad_inner_dim // sp_size, dim=2)

    pad_n = pad_inner_dim // d
    pad_split_n = pad_n // sp_size
    q = q.view(b, seq_len, pad_split_n, d)
    k = k.view(b, seq_len, pad_split_n, d)
    v = v.view(b, seq_len, pad_split_n, d)

    q = rope_apply(q, grid_sizes, freqs)
    k = rope_apply(k, grid_sizes, freqs)

    x = flash_attention(
        q=half(q),
        k=half(k),
        v=half(v),
        k_lens=seq_lens,
        window_size=self.window_size
    )

    # ulysses support
    x = x.flatten(2)
    x = gather_heads_scatter_seq(x, head_dim=2, seq_dim=1)
    if n % sp_size:
        x = unpad_tensor(x, dim=2, unpad_dim_size=seq_len)

    x = self.o(x)
    return x


def ulysses_audio_cross_attn_forward(
    self,
    x,
    audio,
    seq_lens,
    grid_sizes,
    freqs,
    audio_seq_len,
    dtype=torch.bfloat16
):
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    seq_len = seq_lens.max()

    q = self.norm_q(self.q(x))
    k = self.norm_k(self.k(audio))
    v = self.v(audio)

    # ulysses support
    sp_size = get_unified_parallel_world_size()
    if n % sp_size:
        pad_size = sp_size - (n % sp_size)
        pad_size = pad_size * d
        pad_inner_dim = n * d + pad_size
        q = pad_tensor(q, dim=2, padding_size=pad_size)
        k = pad_tensor(k, dim=2, padding_size=pad_size)
        v = pad_tensor(v, dim=2, padding_size=pad_size)
    else:
        pad_inner_dim = n * d

    qq = torch.cat([q, q], dim=2)
    kv = torch.cat([k, v], dim=2)
    qq = gather_seq_scatter_double_head(qq, seq_dim=1, unpadded_dim_size=seq_len)
    kv = gather_seq_scatter_double_head(kv, seq_dim=1, unpadded_dim_size=audio_seq_len)
    q, _ = qq.split(pad_inner_dim // sp_size, dim=2)
    k, v = kv.split(pad_inner_dim // sp_size, dim=2)

    pad_n = pad_inner_dim // d
    pad_split_n = pad_n // sp_size
    q = q.view(b, seq_len, pad_split_n, d)
    k = k.view(b, audio_seq_len, pad_split_n, d)
    v = v.view(b, audio_seq_len, pad_split_n, d)

    hlen_wlen = int(grid_sizes[0][1] * grid_sizes[0][2])
    assert hlen_wlen == 1560 or hlen_wlen == 3600
    q = q.reshape(-1, hlen_wlen, pad_split_n, d)
    k = k.reshape(-1, 16, pad_split_n, d)
    v = v.reshape(-1, 16, pad_split_n, d)

    x = flash_attention(
        q=q,
        k=k,
        v=v,
        k_lens=None,
    )
    x = x.view(b, -1, pad_split_n, d)

    # ulysses support
    x = x.flatten(2)
    x = gather_heads_scatter_seq(x, head_dim=2, seq_dim=1)
    if n % sp_size:
        x = unpad_tensor(x, dim=2, unpad_dim_size=seq_len)

    x = self.o(x)
    return x

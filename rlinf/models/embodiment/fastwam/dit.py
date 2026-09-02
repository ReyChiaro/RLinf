# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Ported from the FastWAM-RL project (https://github.com/.../FastWAM-RL,
# Apache-2.0) at commit e269771 "define the interfaces"; the module content is
# the FastWAM model / flow-matching scheduler definition reused verbatim (only
# imports were made relative and the loguru logger replaced).

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from torch.utils.checkpoint import checkpoint


@dataclass
class PreprocessOutputs:
    hidden_states: torch.Tensor
    freqs: torch.Tensor
    prompt_embeds: torch.Tensor
    prompt_embeds_mask: torch.Tensor
    time_embeds: torch.Tensor
    time_projs: torch.Tensor
    grid_size: tuple[int, ...]
    num_tokens_per_frame: int
    batch_size: int


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """
    Input: [B,H,L,C] with multi-heads
    """
    dtype = x.dtype
    b, h, l, c = x.shape
    x = torch.view_as_complex(x.to(torch.float64).reshape(b, h, l, -1, 2))
    x = torch.view_as_real(x * freqs).flatten(-2)
    return x.to(dtype=dtype)


def get_freqs_cis1d_cache(
    dim: int, end: int = 1024, base: float = 10000.0
) -> torch.Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def get_freqs_cis3d_cache(
    dim: int,
    end: int = 1024,
    base: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    f_freqs_cis = get_freqs_cis1d_cache(dim - 2 * (dim // 3), end, base)
    h_freqs_cis = get_freqs_cis1d_cache(dim // 3, end, base)
    w_freqs_cis = get_freqs_cis1d_cache(dim // 3, end, base)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def sin_embeds1d(
    dim: int, position: torch.Tensor, base: float = 10000.0
) -> torch.Tensor:
    dtype = position.dtype
    # Match the original Wan2.2 implementation: the frequency exponent is
    # computed in float64 regardless of the input dtype, then cast back.
    sinusoid = torch.outer(
        position.to(dtype=torch.float64),
        torch.pow(
            base,
            -torch.arange(dim // 2, device=position.device, dtype=torch.float64)
            / (dim // 2),
        ),
    )
    cis = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return cis.to(dtype=dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize in fp32 exactly like the original Wan2.2 implementation,
        # then cast back: computing ``rsqrt`` in bf16 loses precision that
        # accumulates across the 30 transformer layers.
        dtype = x.dtype
        x = x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype=dtype)


class SelfAttention(nn.Module):
    def __init__(
        self, hidden_dim: int, head_dim: int, num_heads: int, eps: float = 1e-6
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.attn_dim = self.num_heads * self.head_dim

        self.q = nn.Linear(hidden_dim, self.attn_dim)
        self.k = nn.Linear(hidden_dim, self.attn_dim)
        self.v = nn.Linear(hidden_dim, self.attn_dim)
        self.o = nn.Linear(self.attn_dim, hidden_dim)
        self.norm_q = RMSNorm(self.attn_dim, eps=eps)
        self.norm_k = RMSNorm(self.attn_dim, eps=eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs: list[torch.Tensor],
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Input:
            x: [B,L,C]
            freqs: [L,2]
            attn_mask: [L,L]
        """
        b, l, _ = x.shape
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)

        q = q.view(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, l, self.num_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)

        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn = attn.transpose(1, 2).reshape(b, l, self.attn_dim)
        attn = self.o(attn)

        return attn


class CrossAttention(nn.Module):
    def __init__(
        self, hidden_dim: int, head_dim: int, num_heads: int, eps: float = 1e-6
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.attn_dim = self.head_dim * self.num_heads

        self.q = nn.Linear(hidden_dim, self.attn_dim)
        self.k = nn.Linear(hidden_dim, self.attn_dim)
        self.v = nn.Linear(hidden_dim, self.attn_dim)
        self.o = nn.Linear(self.attn_dim, hidden_dim)
        self.norm_q = RMSNorm(self.attn_dim, eps=eps)
        self.norm_k = RMSNorm(self.attn_dim, eps=eps)

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, l, _ = q.shape
        kv_len = kv.shape[1]
        q = self.norm_q(self.q(q))
        k = self.norm_k(self.k(kv))
        v = self.v(kv)

        q = q.view(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, kv_len, self.num_heads, self.head_dim).transpose(1, 2)

        if attn_mask is not None and attn_mask.ndim == 3:
            attn_mask = attn_mask.unsqueeze(1)

        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn = attn.transpose(1, 2).reshape(b, l, self.attn_dim)
        attn = self.o(attn)

        return attn


class GateModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, x: torch.Tensor, gate: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        return x + gate * residual


class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        head_dim: int,
        num_heads: int,
        ffn_dim: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(hidden_dim, head_dim, num_heads, eps=eps)
        self.cross_attn = CrossAttention(hidden_dim, head_dim, num_heads, eps=eps)
        self.norm1 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(hidden_dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, hidden_dim) / hidden_dim**0.5)
        self.gate = GateModule()

    def modulate(
        self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
    ) -> torch.Tensor:
        return (1.0 + scale) * x + shift

    def split_modulation(self, time_embeds: torch.Tensor) -> tuple[torch.Tensor, ...]:
        has_token_dimension = time_embeds.ndim == 4
        chunk_dim = 2 if has_token_dimension else 1
        chunks = (
            self.modulation.to(device=time_embeds.device, dtype=time_embeds.dtype)
            + time_embeds
        ).chunk(6, dim=chunk_dim)
        if has_token_dimension:
            chunks = tuple(chunk.squeeze(2) for chunk in chunks)
        return chunks

    def forward(
        self,
        hidden_states: torch.Tensor,
        prompt_embeds: torch.Tensor,
        time_embeds: torch.Tensor,
        freqs: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Input:
            hidden_states: [B,L,C]
            prompt_embeds: [B,L,C]
            time_embeds: [B,L,6,C]
        """
        msa_shift, msa_scale, msa_gate, mlp_shift, mlp_scale, mlp_gate = (
            self.split_modulation(time_embeds)
        )

        # Gated self-attn
        self_attn = self.self_attn(
            (1.0 + msa_scale) * self.norm1(hidden_states) + msa_shift,
            freqs,
            attn_mask=self_attn_mask,
        )
        hidden_states = hidden_states + msa_gate * self_attn

        # Cross-attn
        hidden_states = hidden_states + self.cross_attn(
            self.norm3(hidden_states),
            prompt_embeds,
            attn_mask=cross_attn_mask,
        )

        # Gated MLP
        mlp = self.ffn((1.0 + mlp_scale) * self.norm2(hidden_states) + mlp_shift)
        hidden_states = hidden_states + mlp_gate * mlp

        return hidden_states


class Head(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        patch_size: tuple[int, int, int],
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(in_dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(in_dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, in_dim) / in_dim**0.5)

    def forward(self, x: torch.Tensor, time_embeds: torch.Tensor) -> torch.Tensor:
        """
        Input:
            time_embeds: [B,L,C]
        """
        # [1,1,2,C] + [B,L,1,C] -> [B,L,2,C] -> [B,L,C] * 2
        shift, scale = (
            self.modulation.unsqueeze(0).to(
                device=time_embeds.device, dtype=time_embeds.dtype
            )
            + time_embeds.unsqueeze(2)
        ).chunk(2, dim=2)
        x = (1.0 + scale.squeeze(2)) * self.norm(x) + shift.squeeze(2)
        x = self.head(x)
        return x


class VideoDiT(ModelMixin, ConfigMixin):
    use_gradient_checkpointing: bool = False

    @register_to_config
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        patch_size: tuple[int, int, int],
        text_dim: int,
        head_dim: int,
        num_heads: int,
        time_dim: int,
        ffn_dim: int,
        num_blocks: int,
        out_dim: int,
        self_attn_mask_mode: Literal[
            "bidirectional", "per_frame_causal", "first_frame_causal"
        ],
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.text_dim = text_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim
        self.self_attn_mask_mode = self_attn_mask_mode

        self.patch_embedding = nn.Conv3d(
            in_channels=in_dim,
            out_channels=hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.time_embedding = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 6 * hidden_dim)
        )

        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_dim, head_dim, num_heads, ffn_dim, eps=eps)
                for _ in range(num_blocks)
            ]
        )
        self.head = Head(hidden_dim, out_dim, patch_size, eps=eps)
        self.freqs = get_freqs_cis3d_cache(head_dim, end=1024, base=10000.0)

    def build_self_attn_mask(
        self,
        video_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        if self.self_attn_mask_mode == "bidirectional":
            return torch.ones(
                (video_seq_len, video_seq_len), device=device, dtype=torch.bool
            )

        if self.self_attn_mask_mode == "per_frame_causal":
            num_frames = video_seq_len // video_tokens_per_frame
            frame_causal = torch.tril(
                torch.ones(num_frames, num_frames, dtype=torch.bool, device=device)
            )
            return frame_causal.repeat_interleave(
                video_tokens_per_frame, dim=0
            ).repeat_interleave(video_tokens_per_frame, dim=1)

        if self.self_attn_mask_mode == "first_frame_causal":
            video_mask = torch.ones(
                (video_seq_len, video_seq_len), dtype=torch.bool, device=device
            )
            first_frame_tokens = min(video_seq_len, video_tokens_per_frame)
            video_mask[:first_frame_tokens, first_frame_tokens:] = False
            return video_mask

        raise ValueError(f"Unsupported self_attn_mask_mode={self.self_attn_mask_mode}.")

    def preprocess(
        self,
        video_tokens: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
    ) -> PreprocessOutputs:
        """
        Input:
            video_tokens: [B,C,T,H,W]
            timestep: [B,]
            prompt_embeds: [B,L,C]
            prompt_embeds_mask: None | [B,L]
        """
        if prompt_embeds_mask is None:
            prompt_embeds_mask = torch.ones(
                (prompt_embeds.shape[0], prompt_embeds.shape[1]),
                device=prompt_embeds.device,
                dtype=torch.bool,
            )
        B, C, T, H, W = video_tokens.shape
        patch_h = self.patch_size[1]
        patch_w = self.patch_size[2]

        assert H % patch_h == 0 and W % patch_w == 0

        tokens_per_frame = (H // patch_h) * (W // patch_w)

        timestep = torch.ones(
            (B, T, tokens_per_frame), device=video_tokens.device
        ) * timestep.view(B, 1, 1)
        # First frame set to 0
        timestep[:, 0, :] = 0
        timestep = timestep.to(dtype=video_tokens.dtype)
        timestep = timestep.flatten(start_dim=1)

        time_embeds = sin_embeds1d(self.time_dim, timestep.flatten())
        time_embeds = self.time_embedding(time_embeds).reshape(B, -1, self.hidden_dim)
        time_projs = self.time_projection(time_embeds).unflatten(
            2, (6, self.hidden_dim)
        )

        video_tokens = self.patch_embedding(video_tokens)
        B, C, T, H, W = video_tokens.shape

        prompt_embeds = self.text_embedding(prompt_embeds)
        prompt_embeds_mask = prompt_embeds_mask.unsqueeze(1).expand(-1, T * H * W, -1)

        video_tokens = video_tokens.permute(0, 2, 3, 4, 1).contiguous()
        video_tokens = video_tokens.view(B, T * H * W, C)

        freqs = (
            torch.cat(
                [
                    self.freqs[0][:T].view(T, 1, 1, -1).expand(T, H, W, -1),
                    self.freqs[1][:H].view(1, H, 1, -1).expand(T, H, W, -1),
                    self.freqs[2][:W].view(1, 1, W, -1).expand(T, H, W, -1),
                ],
                dim=-1,
            )
            .reshape(T * H * W, -1)
            .to(video_tokens.device)
        )

        return PreprocessOutputs(
            hidden_states=video_tokens,
            freqs=freqs,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            time_embeds=time_embeds,
            time_projs=time_projs,
            grid_size=(T, H, W),
            num_tokens_per_frame=tokens_per_frame,
            batch_size=B,
        )

    def postprocess(
        self,
        video_tokens: torch.Tensor,
        time_embeds: torch.Tensor,
        grid_size: tuple[int, int, int],
    ) -> torch.Tensor:
        T, H, W = grid_size
        video_tokens = self.head(video_tokens, time_embeds)
        B, L, C = video_tokens.shape
        video_tokens = video_tokens.view(
            B,
            T,
            H,
            W,
            self.patch_size[0],
            self.patch_size[1],
            self.patch_size[2],
            C // math.prod(self.patch_size),
        )
        video_tokens = video_tokens.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        video_tokens = video_tokens.view(
            B,
            C // math.prod(self.patch_size),
            T * self.patch_size[0],
            H * self.patch_size[1],
            W * self.patch_size[2],
        )
        return video_tokens

    def forward(
        self,
        video_tokens: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
    ) -> torch.Tensor:
        preprocessed_inputs = self.preprocess(
            video_tokens=video_tokens,
            timestep=timestep,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
        )

        video_tokens = preprocessed_inputs.hidden_states
        prompt_embeds = preprocessed_inputs.prompt_embeds
        prompt_embeds_mask = preprocessed_inputs.prompt_embeds_mask
        freqs = preprocessed_inputs.freqs
        time_embeds = preprocessed_inputs.time_embeds
        time_projs = preprocessed_inputs.time_projs
        grid_size = preprocessed_inputs.grid_size

        self_attn_mask = self.build_self_attn_mask(
            video_seq_len=video_tokens.shape[1],
            video_tokens_per_frame=preprocessed_inputs.num_tokens_per_frame,
            device=video_tokens.device,
        )

        for block in self.blocks:
            if self.use_gradient_checkpointing:
                video_tokens = checkpoint(
                    block,
                    use_reentrant=False,
                    hidden_states=video_tokens,
                    prompt_embeds=prompt_embeds,
                    time_embeds=time_projs,
                    freqs=freqs,
                    self_attn_mask=self_attn_mask,
                    cross_attn_mask=prompt_embeds_mask,
                )
            else:
                video_tokens = block(
                    hidden_states=video_tokens,
                    prompt_embeds=prompt_embeds,
                    time_embeds=time_projs,
                    freqs=freqs,
                    self_attn_mask=self_attn_mask,
                    cross_attn_mask=prompt_embeds_mask,
                )

        video_tokens = self.postprocess(video_tokens, time_embeds, grid_size)
        return video_tokens


class ActionDiT(ModelMixin, ConfigMixin):
    use_gradient_checkpointing: bool = False

    @register_to_config
    def __init__(
        self,
        action_dim: int,
        hidden_dim: int,
        text_dim: int,
        head_dim: int,
        num_heads: int,
        time_dim: int,
        ffn_dim: int,
        num_blocks: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.text_dim = text_dim
        self.time_dim = time_dim

        self.action_encoder = nn.Linear(action_dim, hidden_dim)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.time_embedding = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 6 * hidden_dim)
        )

        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_dim, head_dim, num_heads, ffn_dim, eps=eps)
                for _ in range(num_blocks)
            ]
        )
        self.head = nn.Linear(hidden_dim, action_dim)
        self.freqs = get_freqs_cis1d_cache(head_dim, end=1024, base=10000.0)

    def preprocess(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
    ) -> PreprocessOutputs:
        """
        Input:
            action_tokens: [B,T,C]
            timestep: [B,]
        """
        if prompt_embeds_mask is None:
            prompt_embeds_mask = torch.ones(
                (prompt_embeds.shape[0], prompt_embeds.shape[1]),
                device=prompt_embeds.device,
                dtype=torch.bool,
            )
        B, L, C = action_tokens.shape
        timestep = timestep.to(dtype=action_tokens.dtype)
        time_embeds = sin_embeds1d(self.time_dim, timestep)
        time_embeds = self.time_embedding(time_embeds)
        time_projs = self.time_projection(time_embeds).unflatten(
            1, (6, self.hidden_dim)
        )

        action_tokens = self.action_encoder(action_tokens)

        prompt_embeds = self.text_embedding(prompt_embeds)
        prompt_embeds_mask = prompt_embeds_mask.unsqueeze(1).expand(-1, L, -1)

        freqs = self.freqs[:L].to(action_tokens.device)

        return PreprocessOutputs(
            hidden_states=action_tokens,
            freqs=freqs,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            time_embeds=time_embeds,
            time_projs=time_projs,
            grid_size=(L,),
            num_tokens_per_frame=L,
            batch_size=B,
        )

    def postprocess(self, action_tokens: torch.Tensor) -> torch.Tensor:
        return self.head(action_tokens)

    def forward(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
    ) -> torch.Tensor:
        preprocessed_inputs = self.preprocess(
            action_tokens=action_tokens,
            timestep=timestep,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
        )

        action_tokens = preprocessed_inputs.hidden_states
        prompt_embeds = preprocessed_inputs.prompt_embeds
        prompt_embeds_mask = preprocessed_inputs.prompt_embeds_mask
        freqs = preprocessed_inputs.freqs
        time_projs = preprocessed_inputs.time_projs

        for block in self.blocks:
            if self.use_gradient_checkpointing:
                action_tokens = checkpoint(
                    block,
                    use_reentrant=False,
                    hidden_states=action_tokens,
                    prompt_embeds=prompt_embeds,
                    time_embeds=time_projs,
                    freqs=freqs,
                    cross_attn_mask=prompt_embeds_mask,
                )
            else:
                action_tokens = block(
                    hidden_states=action_tokens,
                    prompt_embeds=prompt_embeds,
                    time_embeds=time_projs,
                    freqs=freqs,
                    cross_attn_mask=prompt_embeds_mask,
                )

        action_tokens = self.postprocess(action_tokens)
        return action_tokens

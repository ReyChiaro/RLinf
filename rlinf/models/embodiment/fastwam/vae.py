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

from typing import Any, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

VAE_MEAN = [
    -0.2289,
    -0.0052,
    -0.1323,
    -0.2339,
    -0.2799,
    0.0174,
    0.1838,
    0.1557,
    -0.1382,
    0.0542,
    0.2813,
    0.0891,
    0.1570,
    -0.0098,
    0.0375,
    -0.1825,
    -0.2246,
    -0.1207,
    -0.0698,
    0.5109,
    0.2665,
    -0.2108,
    -0.2158,
    0.2502,
    -0.2055,
    -0.0322,
    0.1109,
    0.1567,
    -0.0729,
    0.0899,
    -0.2799,
    -0.1230,
    -0.0313,
    -0.1649,
    0.0117,
    0.0723,
    -0.2839,
    -0.2083,
    -0.0520,
    0.3748,
    0.0152,
    0.1957,
    0.1433,
    -0.2944,
    0.3573,
    -0.0548,
    -0.1681,
    -0.0667,
]
VAE_STD = [
    0.4765,
    1.0364,
    0.4514,
    1.1677,
    0.5313,
    0.4990,
    0.4818,
    0.5013,
    0.8158,
    1.0344,
    0.5894,
    1.0901,
    0.6885,
    0.6165,
    0.8454,
    0.4978,
    0.5759,
    0.3523,
    0.7135,
    0.6804,
    0.5833,
    1.4146,
    0.8986,
    0.5659,
    0.7069,
    0.5338,
    0.4889,
    0.4917,
    0.4069,
    0.4999,
    0.6866,
    0.4093,
    0.5709,
    0.6065,
    0.6415,
    0.4944,
    0.5726,
    1.2042,
    0.5458,
    1.6887,
    0.3971,
    1.0600,
    0.3943,
    0.5537,
    0.5444,
    0.4089,
    0.7468,
    0.7744,
]
VAE_T_CACHE_SIZE = 2


def get_num_conv3d(model: nn.Module) -> int:
    num = 0
    # This includes DDP module
    for m in model.modules():
        if isinstance(m, CausalConv3D):
            num += 1
    return num


def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Input: [B,C,T,H,W]
    Output: [B,C*p*p,T,H//p,W//p]
    """
    if patch_size == 1:
        return x
    b, c, t, h, w = x.shape
    x = x.view(b, c, t, h // patch_size, patch_size, w // patch_size, patch_size)
    # Wan stores each spatial patch in (width, height) order: (c r q).
    x = x.permute(0, 1, 6, 4, 2, 3, 5).contiguous()
    x = x.view(b, c * patch_size * patch_size, t, h // patch_size, w // patch_size)
    return x


def unpatchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Input: [B,C*p*p,T,H//p,W//p]
    Output: [B,C,T,H,W]
    """
    if patch_size == 1:
        return x
    b, c, t, h, w = x.shape
    # The packed channel order is (c r q), matching patchify and Wan2.2.
    x = x.view(b, c // patch_size // patch_size, patch_size, patch_size, t, h, w)
    x = x.permute(0, 1, 4, 5, 3, 6, 2).contiguous()
    x = x.view(b, c // patch_size // patch_size, t, h * patch_size, w * patch_size)
    return x


def handle_cached_causal_conv3d(
    module: CausalConv3D,
    x: torch.Tensor,
    feature_cache: list[torch.Tensor],
    feature_idx: list[int],
) -> tuple[torch.Tensor, list[torch.Tensor], list[int]]:
    idx = feature_idx[0]
    cache_x = x[:, :, -VAE_T_CACHE_SIZE:, :, :].clone()
    if cache_x.shape[2] < VAE_T_CACHE_SIZE and feature_cache[idx] is not None:
        cache_x = torch.cat(
            [
                feature_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                cache_x,
            ],
            dim=2,
        )
    x = module(x, feature_cache=feature_cache[idx])
    feature_cache[idx] = cache_x
    feature_idx[0] += 1
    return x, feature_cache, feature_idx


class CausalConv3D(nn.Conv3d):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        # Handle [B,C,T,H,W], self.padding is the original Conv3d padding
        self.causal_padding = (
            self.padding[2],
            self.padding[2],
            self.padding[1],
            self.padding[1],
            2 * self.padding[0],
            0,
        )
        # Disable self.padding
        self.padding = (0, 0, 0)

    def forward(
        self, x: torch.Tensor, feature_cache: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        padding = list(self.causal_padding)
        # Use cached features or zeros to pad
        if feature_cache is not None and self.causal_padding[4] > 0:
            feature_cache = feature_cache.to(x.device)
            x = torch.cat([feature_cache, x], dim=2)
            padding[4] -= feature_cache.shape[2]
        return super().forward(F.pad(x, pad=padding))


class VAERMSNorm(nn.Module):
    def __init__(self, dim: int, bias: bool = False, images: bool = False) -> None:
        super().__init__()

        # Residual blocks normalize 5D video tensors, while AttentionBlock
        # normalizes per-frame 4D tensors.  These shapes match the official
        # Wan2.2 VAE checkpoint exactly.
        shape = (dim, 1, 1) if images else (dim, 1, 1, 1)
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=1) * self.scale * self.gamma + self.bias


class AvgDown3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor_t: int,  # Temporal
        factor_s: int,  # Spatial
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s

        assert self.in_channels * self.factor % self.out_channels == 0
        self.group_size = self.in_channels * self.factor // self.out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: [B,Cin,T,H,W]
        Middle: [B,Cin*ft*fs*fs,G,T//ft,H//fs,W//fs]
        Output: [B,Cout,T//ft,H//fs,W//fs]
        """
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        x = F.pad(x, (0, 0, 0, 0, pad_t, 0))

        b, c, t, h, w = x.shape
        x = x.view(
            b,
            c,
            t // self.factor_t,
            self.factor_t,
            h // self.factor_s,
            self.factor_s,
            w // self.factor_s,
            self.factor_s,
        )
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(
            b,
            c * self.factor,
            t // self.factor_t,
            h // self.factor_s,
            w // self.factor_s,
        )
        x = x.view(
            b,
            self.out_channels,
            self.group_size,
            t // self.factor_t,
            h // self.factor_s,
            w // self.factor_s,
        )
        x = x.mean(dim=2)
        return x


class DupUp3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor_t: int,
        factor_s: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s

        assert self.out_channels * self.factor % self.in_channels == 0
        self.num_repeats = out_channels * self.factor // self.in_channels

    def forward(self, x: torch.Tensor, first_chunk: bool = False) -> torch.Tensor:
        """
        Input: [B,Cin,T,H,W]
        """
        x = x.repeat_interleave(self.num_repeats, dim=1)
        b, c, t, h, w = x.shape
        x = x.view(
            b,
            self.out_channels,
            self.factor_t,
            self.factor_s,
            self.factor_s,
            t,
            h,
            w,
        )
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(
            b,
            self.out_channels,
            t * self.factor_t,
            h * self.factor_s,
            w * self.factor_s,
        )
        if first_chunk:
            x = x[:, :, self.factor_t - 1 :, :, :]
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.residual = nn.Sequential(
            VAERMSNorm(self.in_dim, bias=False),
            nn.SiLU(),
            CausalConv3D(self.in_dim, self.out_dim, kernel_size=3, padding=1),
            VAERMSNorm(self.out_dim, bias=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            CausalConv3D(self.out_dim, self.out_dim, kernel_size=3, padding=1),
        )
        self.shortcut = (
            CausalConv3D(self.in_dim, self.out_dim, kernel_size=1)
            if self.in_dim != self.out_dim
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        feature_cache: Optional[list[torch.Tensor]] = None,
        feature_idx: list[int] = [0],
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[int]]:
        short = self.shortcut(x)
        for layer in self.residual:
            if isinstance(layer, CausalConv3D) and feature_cache is not None:
                x, feature_cache, feature_idx = handle_cached_causal_conv3d(
                    module=layer,
                    x=x,
                    feature_cache=feature_cache,
                    feature_idx=feature_idx,
                )
            else:
                x = layer(x)
        return x + short, feature_cache, feature_idx


class AttentionBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()

        self.dim = dim
        self.norm = VAERMSNorm(dim, bias=False, images=True)
        self.to_qkv = nn.Conv2d(in_channels=dim, out_channels=3 * dim, kernel_size=1)
        self.proj = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1)

        nn.init.zeros_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        b, c, t, h, w = x.shape

        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(b * t, c, h, w)
        x = self.norm(x)

        qkv = self.to_qkv(x)
        q, k, v = (
            qkv.reshape(b * t, 1, c * 3, h * w)
            .permute(0, 1, 3, 2)
            .contiguous()
            .chunk(3, dim=-1)
        )

        x = F.scaled_dot_product_attention(q, k, v)
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)
        x = self.proj(x)

        x = x.view(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()
        return x + residual


class Resample(nn.Module):
    MODE = ["none", "upsample2d", "upsample3d", "downsample2d", "downsample3d"]

    def __init__(
        self,
        dim: int,
        mode: Literal[
            "none", "upsample2d", "upsample3d", "downsample2d", "downsample3d"
        ],
    ) -> None:
        assert mode in self.MODE, (
            f"Resample mode must be one of [{self.MODE}], got `{mode}`."
        )
        super().__init__()

        self.dim = dim
        self.mode = mode

        if self.mode in ["upsample2d", "upsample3d"]:
            # The `mode` arg will be passed into F.interpolate
            self.resample = nn.Sequential(
                nn.Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            )
            if self.mode == "upsample3d":
                self.time_conv = CausalConv3D(
                    dim,
                    dim * 2,
                    kernel_size=(3, 1, 1),
                    padding=(1, 0, 0),
                )

        elif self.mode in ["downsample2d", "downsample3d"]:
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, kernel_size=3, stride=2),
            )
            if self.mode == "downsample3d":
                self.time_conv = CausalConv3D(
                    dim,
                    dim,
                    kernel_size=(3, 1, 1),
                    stride=(2, 1, 1),
                    padding=(0, 0, 0),
                )
        else:
            self.resample = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        feature_cache: Optional[list[torch.Tensor]] = None,
        feature_idx: list[int] = [0],
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[int]]:
        b, c, t, h, w = x.shape

        if self.mode == "upsample3d" and feature_cache is not None:
            idx = feature_idx[0]
            if feature_cache[idx] is None:
                feature_cache[idx] = "Rep"
                feature_idx[0] += 1
            else:
                cache_x = x[:, :, -VAE_T_CACHE_SIZE:, :, :].clone()
                if (
                    cache_x.shape[2] < VAE_T_CACHE_SIZE
                    and feature_cache[idx] is not None
                ):
                    if feature_cache[idx] != "Rep":
                        cache_x = torch.cat(
                            [
                                feature_cache[idx][:, :, -1, :, :]
                                .unsqueeze(2)
                                .to(cache_x.device),
                                cache_x,
                            ],
                            dim=2,
                        )
                    else:
                        cache_x = torch.cat(
                            [
                                torch.zeros_like(cache_x).to(cache_x.device),
                                cache_x,
                            ],
                            dim=2,
                        )

                if feature_cache[idx] == "Rep":
                    x = self.time_conv(x)
                else:
                    x = self.time_conv(x, feature_cache[idx])
                feature_cache[idx] = cache_x
                feature_idx[0] += 1

                x = x.reshape(b, 2, c, t, h, w)
                x = torch.stack([x[:, 0, ...], x[:, 1, ...]], dim=3)
                x = x.reshape(b, c, t * 2, h, w)

        t = x.shape[2]
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(b * t, c, h, w)
        x = self.resample(x)
        _, c, h, w = x.shape
        x = x.view(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()

        if self.mode == "downsample3d" and feature_cache is not None:
            idx = feature_idx[0]
            if feature_cache[idx] is None:
                feature_cache[idx] = x.clone()
                feature_idx[0] += 1
            else:
                cache_x = x[:, :, -1:, :, :].clone()
                x = self.time_conv(
                    torch.cat([feature_cache[idx][:, :, -1:, :, :], x], dim=2)
                )
                feature_cache[idx] = cache_x
                feature_idx[0] += 1

        return x, feature_cache, feature_idx


class DownResidualBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        temporal_downsample: bool,
        spatial_downsample: bool,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.avg_shortcut = AvgDown3D(
            in_channels=in_dim,
            out_channels=out_dim,
            factor_t=2 if temporal_downsample else 1,
            factor_s=2 if spatial_downsample else 1,
        )

        downsamples = []
        for _ in range(num_res_blocks):
            downsamples.append(
                ResidualBlock(in_dim=in_dim, out_dim=out_dim, dropout=dropout)
            )
            in_dim = out_dim

        if spatial_downsample:
            mode = "downsample3d" if temporal_downsample else "downsample2d"
            downsamples.append(Resample(dim=out_dim, mode=mode))

        self.downsamples = nn.Sequential(*downsamples)

    def forward(
        self,
        x: torch.Tensor,
        feature_cache: Optional[list[torch.Tensor]] = None,
        feature_idx: list[int] = [0],
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[int]]:
        short = self.avg_shortcut(x)
        for layer in self.downsamples:
            x, feature_cache, feature_idx = layer(x, feature_cache, feature_idx)
        return x + short, feature_cache, feature_idx


class UpResidualBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        temporal_upsample: bool,
        spatial_upsample: bool,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if spatial_upsample:
            self.avg_shortcut = DupUp3D(
                in_channels=in_dim,
                out_channels=out_dim,
                factor_t=2 if temporal_upsample else 1,
                factor_s=2 if spatial_upsample else 1,
            )
        else:
            self.avg_shortcut = None

        upsamples = []
        for _ in range(num_res_blocks):
            upsamples.append(
                ResidualBlock(in_dim=in_dim, out_dim=out_dim, dropout=dropout)
            )
            in_dim = out_dim

        if spatial_upsample:
            mode = "upsample3d" if temporal_upsample else "upsample2d"
            upsamples.append(Resample(dim=out_dim, mode=mode))

        self.upsamples = nn.Sequential(*upsamples)

    def forward(
        self,
        x: torch.Tensor,
        feature_cache: Optional[list[torch.Tensor]] = None,
        feature_idx: list[int] = [0],
        first_chunk: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[int]]:
        short = None
        if self.avg_shortcut is not None:
            short = self.avg_shortcut(x, first_chunk=first_chunk)
        for layer in self.upsamples:
            x, feature_cache, feature_idx = layer(x, feature_cache, feature_idx)
        if short is None:
            return x, feature_cache, feature_idx
        return x + short, feature_cache, feature_idx


class VAEEncoder3D(nn.Module):
    def __init__(
        self,
        dim: int,
        z_dim: int,
        dim_mults: list[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        attn_scales: list[float] = [],
        temporal_downsample: list[bool] = [False, True, True],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mults = dim_mults
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temporal_downsample = temporal_downsample
        self.dropout = dropout

        dims = [dim * m for m in [1] + self.dim_mults]

        self.conv1 = CausalConv3D(
            in_channels=12,
            out_channels=dims[0],
            kernel_size=3,
            padding=1,
        )

        # 8x spatial downsample + 4x temporal downsample
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_downsample = (
                self.temporal_downsample[i]
                if i < len(self.temporal_downsample)
                else False
            )
            s_downsample = i != len(self.dim_mults) - 1
            downsamples.append(
                DownResidualBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_res_blocks=self.num_res_blocks,
                    temporal_downsample=t_downsample,
                    spatial_downsample=s_downsample,
                    dropout=dropout,
                )
            )
        self.downsamples = nn.Sequential(*downsamples)

        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout),
            AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout),
        )

        self.head = nn.Sequential(
            VAERMSNorm(out_dim, bias=False),
            nn.SiLU(),
            CausalConv3D(
                in_channels=out_dim, out_channels=z_dim, kernel_size=3, padding=1
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        feature_cache: Optional[list[torch.Tensor]] = None,
        feature_idx: list[int] = [0],
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[int]]:
        if feature_cache is not None:
            x, feature_cache, feature_idx = handle_cached_causal_conv3d(
                module=self.conv1,
                x=x,
                feature_cache=feature_cache,
                feature_idx=feature_idx,
            )
        else:
            x = self.conv1(x)

        for layer in self.downsamples:
            if feature_cache is not None:
                x, feature_cache, feature_idx = layer(x, feature_cache, feature_idx)
            else:
                x = layer(x)

        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feature_cache is not None:
                x, feature_cache, feature_idx = layer(x, feature_cache, feature_idx)
            else:
                x = layer(x)

        for layer in self.head:
            if isinstance(layer, CausalConv3D) and feature_cache is not None:
                x, feature_cache, feature_idx = handle_cached_causal_conv3d(
                    module=layer,
                    x=x,
                    feature_cache=feature_cache,
                    feature_idx=feature_idx,
                )
            else:
                x = layer(x)

        return x, feature_cache, feature_idx


class VAEDecoder3D(nn.Module):
    def __init__(
        self,
        dim: int,
        z_dim: int,
        dim_mults: list[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        attn_scales: list[float] = [],
        temporal_upsample: list[bool] = [True, True, False],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mults = dim_mults
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temporal_upsample = temporal_upsample
        self.dropout = dropout

        dims = [dim * m for m in [dim_mults[-1]] + dim_mults[::-1]]

        self.conv1 = CausalConv3D(z_dim, dims[0], kernel_size=3, padding=1)

        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout),
            AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout),
        )

        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_upsample = (
                self.temporal_upsample[i] if i < len(self.temporal_upsample) else False
            )
            s_upsample = i != len(dims[:-1]) - 1
            upsamples.append(
                UpResidualBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_res_blocks=num_res_blocks + 1,
                    temporal_upsample=t_upsample,
                    spatial_upsample=s_upsample,
                    dropout=dropout,
                )
            )
        self.upsamples = nn.Sequential(*upsamples)

        self.head = nn.Sequential(
            VAERMSNorm(dim=out_dim, bias=False),
            nn.SiLU(),
            CausalConv3D(
                in_channels=out_dim, out_channels=12, kernel_size=3, padding=1
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        feature_cache: Optional[list[torch.Tensor]] = None,
        feature_idx: list[int] = [0],
        first_chunk: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[int]]:
        if feature_cache is not None:
            x, feature_cache, feature_idx = handle_cached_causal_conv3d(
                module=self.conv1,
                x=x,
                feature_cache=feature_cache,
                feature_idx=feature_idx,
            )
        else:
            x = self.conv1(x)

        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feature_cache is not None:
                x, feature_cache, feature_idx = layer(x, feature_cache, feature_idx)
            else:
                x = layer(x)

        for layer in self.upsamples:
            if feature_cache is not None:
                x, feature_cache, feature_idx = layer(
                    x,
                    feature_cache,
                    feature_idx,
                    first_chunk=first_chunk,
                )
            else:
                x = layer(x)

        for layer in self.head:
            if isinstance(layer, CausalConv3D) and feature_cache is not None:
                x, feature_cache, feature_idx = handle_cached_causal_conv3d(
                    module=layer,
                    x=x,
                    feature_cache=feature_cache,
                    feature_idx=feature_idx,
                )
            else:
                x = layer(x)

        return x, feature_cache, feature_idx


class VAEModel(nn.Module):
    def __init__(
        self,
        encode_dim: int,
        z_dim: int,
        decode_dim: int,
        dim_mults: list[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        attn_scales: list[float] = [],
        temporal_downsample: list[bool] = [False, True, True],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encode_dim = encode_dim
        self.z_dim = z_dim
        self.decode_dim = decode_dim
        self.dim_mults = dim_mults
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.dropout = dropout
        self.temporal_downsample = temporal_downsample
        self.temporal_upsample = temporal_downsample[::-1]

        self.encoder = VAEEncoder3D(
            dim=encode_dim,
            z_dim=2 * z_dim,
            dim_mults=dim_mults,
            num_res_blocks=num_res_blocks,
            attn_scales=attn_scales,
            temporal_downsample=self.temporal_downsample,
            dropout=dropout,
        )

        self.conv1 = CausalConv3D(2 * z_dim, 2 * z_dim, kernel_size=1)
        self.conv2 = CausalConv3D(z_dim, z_dim, kernel_size=1)

        self.decoder = VAEDecoder3D(
            dim=decode_dim,
            z_dim=z_dim,
            dim_mults=dim_mults,
            num_res_blocks=num_res_blocks,
            attn_scales=attn_scales,
            temporal_upsample=self.temporal_upsample,
            dropout=dropout,
        )

        # Register cache
        self.clear_cache()

    def clear_cache(self) -> None:
        # Encoder
        self._num_encode_conv = get_num_conv3d(self.encoder)
        self._encode_conv_idx = [0]
        self._encode_feat_map = [None] * self._num_encode_conv

        # Decoder
        self._num_decode_conv = get_num_conv3d(self.decoder)
        self._decode_conv_idx = [0]
        self._decode_feat_map = [None] * self._num_decode_conv

    def encode(self, x: torch.Tensor, scales: list[torch.Tensor]) -> torch.Tensor:
        """
        Input:
            x: [B,C,T,H,W],
            scales: list of 2 tensors
        """
        self.clear_cache()

        # [B,3,T,H,W] -> [B,12,T,H//2,W//2]
        x = patchify(x, patch_size=2)
        T = x.shape[2]
        num_frames = 1 + (T - 1) // 4

        for f in range(num_frames):
            self._encode_conv_idx = [0]

            if f == 0:
                out, self._encode_feat_map, self._encode_conv_idx = self.encoder(
                    x[:, :, :1, :, :],
                    feature_cache=self._encode_feat_map,
                    feature_idx=self._encode_conv_idx,
                )
            else:
                other_out, self._encode_feat_map, self._encode_conv_idx = self.encoder(
                    x[:, :, 1 + 4 * (f - 1) : 1 + 4 * f, :, :],
                    feature_cache=self._encode_feat_map,
                    feature_idx=self._encode_conv_idx,
                )
                out = torch.cat([out, other_out], dim=2)

        mu, log_var = self.conv1(out).chunk(2, dim=1)
        scales = [s.to(device=mu.device, dtype=mu.dtype) for s in scales]
        mu = (mu - scales[0].view(1, self.z_dim, 1, 1, 1)) * scales[1].view(
            1, self.z_dim, 1, 1, 1
        )

        self.clear_cache()
        return mu

    def decode(self, x: torch.Tensor, scales: list[torch.Tensor]) -> torch.Tensor:
        self.clear_cache()
        scales = [s.to(device=x.device, dtype=x.dtype) for s in scales]
        x = x / scales[1].view(1, self.z_dim, 1, 1, 1) + scales[0].view(
            1, self.z_dim, 1, 1, 1
        )

        T = x.shape[2]
        x = self.conv2(x)

        for f in range(T):
            self._decode_conv_idx = [0]

            if f == 0:
                out, self._decode_feat_map, self._decode_conv_idx = self.decoder(
                    x[:, :, f : f + 1, :, :],
                    feature_cache=self._decode_feat_map,
                    feature_idx=self._decode_conv_idx,
                    first_chunk=True,
                )
            else:
                other_out, self._decode_feat_map, self._decode_conv_idx = self.decoder(
                    x[:, :, f : f + 1, :, :],
                    feature_cache=self._decode_feat_map,
                    feature_idx=self._decode_conv_idx,
                    first_chunk=False,
                )
                out = torch.cat([out, other_out], dim=2)
        out = unpatchify(out, patch_size=2)

        self.clear_cache()
        return out


class WanVideoVAE(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        encode_dim: int = 160,
        z_dim: int = 48,
        decode_dim: int = 256,
        dim_mults: list[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        temporal_downsample: list[bool] = [False, True, True],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.mean = torch.tensor(VAE_MEAN)
        self.std = torch.tensor(VAE_STD)

        self.scales = [self.mean, 1.0 / self.std]

        self.model = VAEModel(
            encode_dim=encode_dim,
            z_dim=z_dim,
            decode_dim=decode_dim,
            dim_mults=dim_mults,
            num_res_blocks=num_res_blocks,
            temporal_downsample=temporal_downsample,
            dropout=dropout,
        )

        self.upsampling_factor = 16
        self.temporal_downsample_factor = 4
        self.z_dim = z_dim

    def single_encode(self, video: torch.Tensor, device: torch.device) -> torch.Tensor:
        video = video.to(device)
        x = self.model.encode(video, self.scales)
        return x

    def single_decode(
        self, hidden_states: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        hidden_states = hidden_states.to(device)
        video = self.model.decode(hidden_states, self.scales)
        return video.clamp_(-1, 1)

    def encode(
        self, videos: list[torch.Tensor] | torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        hidden_states = []
        for video in videos:
            hidden_state = self.single_encode(video.unsqueeze(0), device)
            hidden_states.append(hidden_state)
        return torch.cat(hidden_states, dim=0)

    def decode(
        self, hidden_states: list[torch.Tensor] | torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        videos = []
        for hidden in hidden_states:
            video = self.single_decode(hidden.unsqueeze(0), device)
            videos.append(video.squeeze(0))
        videos = torch.stack(videos)
        return videos

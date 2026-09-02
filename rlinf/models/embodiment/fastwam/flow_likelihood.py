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

"""Torch-only math helpers shared by the FastWAM RL wrapper and its tests.

These functions encode the Flow-GRPO action-flow quantities:

* min-max normalization to ``[-1, 1]`` of raw actions / proprio states
  (:func:`_normalize_minmax` / :func:`_denormalize_minmax`);
* the reverse-SDE transition ``mean``/``std`` of the flow sampler
  (:func:`transition_mean_std`, mirroring the ported
  ``FlowMatchingScheduler.stochastic_step``);
* the per-element Gaussian log-probability of an observed transition
  (:func:`elementwise_gaussian_log_prob`).

The module intentionally imports nothing beyond ``torch`` so it can be unit
tested without the FastWAM (diffusers/transformers) runtime.
"""

from __future__ import annotations

import math

import torch

_LOG_2PI = math.log(2.0 * math.pi)


def normalize_minmax(
    values: torch.Tensor,
    minimum: torch.Tensor,
    maximum: torch.Tensor,
) -> torch.Tensor:
    """Map ``[minimum, maximum]`` to ``[-1, 1]`` (clamped to ``[-5, 5]``)."""
    value_range = (maximum - minimum).clamp_min(1e-4)
    scale = 2.0 / value_range
    offset = -1.0 - scale * minimum
    return (values * scale + offset).clamp(-5.0, 5.0)


def denormalize_minmax(
    values: torch.Tensor,
    minimum: torch.Tensor,
    maximum: torch.Tensor,
) -> torch.Tensor:
    """Inverse of :func:`normalize_minmax`."""
    value_range = (maximum - minimum).clamp_min(1e-4)
    scale = 2.0 / value_range
    offset = -1.0 - scale * minimum
    return (values - offset) / scale


def transition_mean_std(
    sample: torch.Tensor,
    velocity: torch.Tensor,
    sigma: torch.Tensor,
    next_sigma: torch.Tensor,
    noise_level: float,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reverse-SDE transition ``mean``/``std`` of the Flow-GRPO sampler.

    ``sigma``/``next_sigma`` may be scalars or per-batch-row tensors; the
    ``[B, T, C]`` layout of ``sample``/``velocity`` is preserved by the mean
    and std tensors.
    """
    xt = sample.to(dtype=torch.float32)
    vt = velocity.to(dtype=torch.float32)
    sigma = sigma.to(dtype=torch.float32)
    next_sigma = next_sigma.to(dtype=torch.float32)
    while sigma.ndim < sample.ndim:
        sigma = sigma.unsqueeze(-1)
        next_sigma = next_sigma.unsqueeze(-1)
    delta = next_sigma - sigma
    effective_sigma = torch.where(sigma >= 1.0 - eps, next_sigma, sigma)
    denominator = (1.0 - effective_sigma).clamp_min(eps)
    diffusion = torch.sqrt(sigma / denominator) * noise_level
    mean = (
        xt * (1.0 + diffusion.square() / (2.0 * sigma.clamp_min(eps)) * delta)
        + vt
        * (1.0 + diffusion.square() * (1.0 - sigma) / (2.0 * sigma.clamp_min(eps)))
        * delta
    )
    std = diffusion * torch.sqrt((-delta).clamp_min(eps))
    return mean, std


def elementwise_gaussian_log_prob(
    next_sample: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Per-element Gaussian log-probability of an observed SDE transition.

    ``next_sample`` is detached so gradients flow through the predicted
    transition ``mean`` only (the Flow-GRPO likelihood surrogate). The result
    keeps the action layout ``[B, T, C]``; RLinf's ``chunk_level`` loss
    preprocessing sums over ``T x C``.
    """
    next_sample = next_sample.detach()
    std = std.clamp_min(1e-6)
    log_prob = -0.5 * ((next_sample - mean) / std).square()
    log_prob = log_prob - torch.log(std) - 0.5 * _LOG_2PI
    return log_prob.to(dtype=torch.float32)

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

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin


@dataclass
class FlowStates:
    sample: torch.Tensor
    sigma: torch.Tensor
    next_sigma: torch.Tensor
    log_prob: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor


class FlowMatchingScheduler(SchedulerMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        num_training_steps: int = 1000,
        shift: float = 5.0,
        min_training_loss: float = math.exp(-0.5),
        eps: float = 1e-6,
    ):
        self.num_training_steps = num_training_steps
        self.shift = shift
        self.min_training_loss = min_training_loss
        self.eps = eps

        u = torch.linspace(0, 1, 4097, dtype=torch.float64)
        sigma = self.shift_sigma(u)
        weight = torch.exp(-2 * (sigma - 0.5).square()) - self.min_training_loss
        self.norm_training_weight = float(torch.trapezoid(weight, u).item())

    def shift_sigma(self, u: torch.Tensor) -> torch.Tensor:
        return self.shift * u / (1.0 + (self.shift - 1) * u)

    def sample_training_sigmas(
        self,
        batch_size: int,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        u = torch.rand(
            batch_size, device=device, dtype=torch.float32, generator=generator
        )
        return self.shift_sigma(u)

    def sample_inference_sigmas(
        self, num_inference_steps: int, device: torch.device
    ) -> torch.Tensor:
        ts = torch.linspace(
            1, 0, num_inference_steps + 1, device=device, dtype=torch.float32
        )
        return self.shift_sigma(ts)

    def convert_to_model_timesteps(self, sigmas: torch.Tensor) -> torch.Tensor:
        dtype = sigmas.dtype
        return (sigmas * float(self.num_training_steps)).to(dtype=dtype)

    def add_noise(
        self, xt: torch.Tensor, noise: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        dtype = xt.dtype
        sigma = sigma.float()
        while sigma.ndim < xt.ndim:
            sigma = sigma.unsqueeze(-1)
        xt = (1.0 - sigma) * xt.float() + sigma * noise.float()
        return xt.to(dtype=dtype)

    def training_target(
        self, original_sample: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        return noise - original_sample

    def training_weight(self, sigma: torch.Tensor) -> torch.Tensor:
        sigma = sigma.to(dtype=torch.float32)
        weight = torch.exp(-2 * (sigma - 0.5) ** 2) - self.min_training_loss
        return weight / (self.norm_training_weight + self.eps)

    def step(
        self,
        xt: torch.Tensor,
        v: torch.Tensor,
        sigma: torch.Tensor,
        next_sigma: torch.Tensor,
    ) -> torch.Tensor:
        dtype = xt.dtype
        delta_sigma = (next_sigma - sigma).float()
        while delta_sigma.ndim < xt.ndim:
            delta_sigma = delta_sigma.unsqueeze(-1)
        xt = xt.float() + delta_sigma * v.float()
        return xt.to(dtype=dtype)

    def stochastic_step(
        self,
        sample: torch.Tensor,
        velocity: torch.Tensor,
        sigma: torch.Tensor,
        next_sigma: torch.Tensor,
        noise_level: float,
        next_sample: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> FlowStates:
        """One reverse-SDE transition and its per-sample log probability."""

        dtype = sample.dtype
        xt, vt = sample.float(), velocity.float()
        sigma, next_sigma = sigma.float(), next_sigma.float()

        while sigma.ndim < sample.ndim:
            sigma = sigma.unsqueeze(-1)
            next_sigma = next_sigma.unsqueeze(-1)
        delta = next_sigma - sigma

        # sigma=1 is the pure-noise endpoint.  The reverse-SDE coefficient is
        # singular there, so the first transition uses the next schedule value
        # as the effective endpoint, matching the Flow-GRPO sampler derivation.
        effective_sigma = torch.where(sigma >= 1.0 - self.eps, next_sigma, sigma)
        denominator = (1.0 - effective_sigma).clamp_min(self.eps)
        diffusion = torch.sqrt(sigma / denominator) * noise_level

        mean = xt * (1 + diffusion.square() / (2 * sigma.clamp_min(self.eps)) * delta)
        mean = (
            mean
            + vt
            * (1 + diffusion.square() * (1 - sigma) / (2 * sigma.clamp_min(self.eps)))
            * delta
        )
        std = diffusion * torch.sqrt((-delta).clamp_min(self.eps))

        if next_sample is None:
            noise = torch.randn(
                sample.shape,
                device=sample.device,
                dtype=torch.float32,
                generator=generator,
            )
            next_xt = mean + std * noise
        else:
            next_xt = next_sample.float()

        log_prob = -0.5 * ((next_xt.detach() - mean) / std).square()
        log_prob = log_prob - torch.log(std) - 0.5 * math.log(2 * math.pi)
        log_prob = log_prob.flatten(1).mean(dim=1)

        return FlowStates(
            sample=next_xt.to(dtype),
            sigma=sigma,
            next_sigma=next_sigma,
            log_prob=log_prob,
            mean=mean,
            std=std,
        )

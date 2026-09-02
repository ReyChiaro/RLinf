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

"""Tests for the torch-only Flow-GRPO likelihood helpers of FastWAM."""

import numpy as np
import torch

from rlinf.models.embodiment.fastwam.flow_likelihood import (
    denormalize_minmax,
    elementwise_gaussian_log_prob,
    normalize_minmax,
    transition_mean_std,
)


def test_normalize_denormalize_roundtrip():
    values = torch.tensor([[0.0, 1.0], [0.5, -1.0]], dtype=torch.float32)
    minimum = torch.tensor([-1.0, -2.0], dtype=torch.float32)
    maximum = torch.tensor([1.0, 2.0], dtype=torch.float32)
    normalized = normalize_minmax(values, minimum, maximum)
    # Each column maps [min, max] -> [-1, 1]; the symmetric [-1, 1] column is
    # the identity and the [-2, 2] column halves the input values.
    torch.testing.assert_close(
        normalized, torch.tensor([[0.0, 0.5], [0.5, -0.5]], dtype=torch.float32)
    )
    restored = denormalize_minmax(normalized, minimum, maximum)
    torch.testing.assert_close(restored, values)


def test_transition_mean_std_matches_flow_scheduler_formula():
    # Reference formula of FastWAM's FlowMatchingScheduler.stochastic_step for a
    # single sample (deterministic values, noise_level=0.5).
    torch.manual_seed(0)
    sample = torch.randn(2, 4, 3, dtype=torch.float32)
    velocity = torch.randn_like(sample)
    sigma = torch.full((2,), 0.7)
    next_sigma = torch.full((2,), 0.4)
    mean, std = transition_mean_std(sample, velocity, sigma, next_sigma, 0.5)

    xt = sample.float()
    vt = velocity.float()
    delta = 0.4 - 0.7
    diffusion = torch.sqrt(torch.tensor(0.7) / (1.0 - 0.7)) * 0.5
    expected_mean = (
        xt * (1 + diffusion.square() / (2 * 0.7) * delta)
        + vt * (1 + diffusion.square() * (1 - 0.7) / (2 * 0.7)) * delta
    )
    expected_std = torch.full_like(
        std, float(diffusion * torch.sqrt(torch.tensor(-delta)))
    )
    torch.testing.assert_close(mean, expected_mean)
    torch.testing.assert_close(std, expected_std)


def test_elementwise_log_prob_and_gradient():
    torch.manual_seed(1)
    sample = torch.randn(1, 4, 3, dtype=torch.float32)
    velocity = torch.randn_like(sample, requires_grad=True)
    sigma = torch.tensor([0.6])
    next_sigma = torch.tensor([0.3])
    mean, std = transition_mean_std(sample, velocity, sigma, next_sigma, 0.7)
    next_sample = mean + std * torch.randn_like(mean)
    log_prob = elementwise_gaussian_log_prob(next_sample, mean, std)

    assert log_prob.shape == (1, 4, 3)
    assert log_prob.dtype == torch.float32
    # The gradient flows to the velocity network only through the mean.
    assert velocity.grad is None
    log_prob.sum().backward()
    assert velocity.grad is not None
    assert torch.isfinite(velocity.grad).all()


def test_high_log_prob_at_drawn_transition():
    torch.manual_seed(2)
    sample = torch.randn(1, 4, 3, dtype=torch.float32)
    velocity = torch.zeros_like(sample)
    sigma = torch.tensor([0.5])
    next_sigma = torch.tensor([0.25])
    mean, std = transition_mean_std(sample, velocity, sigma, next_sigma, 0.0)
    # noise_level=0 -> deterministic transition (std ~ 0) is excluded by design;
    # here we still require the formula to be defined.
    assert (std >= 0).all()
    log_prob = elementwise_gaussian_log_prob(mean, mean, torch.ones_like(std))
    expected = np.log(1.0 / np.sqrt(2.0 * np.pi))
    torch.testing.assert_close(log_prob, torch.full_like(log_prob, expected))

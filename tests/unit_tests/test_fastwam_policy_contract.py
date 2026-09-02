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

"""Contract tests for the FastWAM RL wrapper with a fake velocity network.

The tests exercise ``FastWAMActionPolicy`` (rollout sampling, cached
``forward_inputs`` and the actor-side likelihood recompute) without loading any
real checkpoint: the FastWAM shell is replaced by a plain stub whose action
flow uses a learnable scalar velocity field. This keeps the test fast while
validating the Flow-GRPO plumbing (prev_logprobs layout, on-policy recompute
consistency, gradients through the action expert).

The FastWAM/diffusers imports are optional; the tests skip when those
dependencies are not installed (mirrors other optional-dependency unit tests).
"""

import pytest
import torch
import torch.nn as nn

ones = torch.ones(7)


class _FakeWAM(nn.Module):
    """FastWAM shell with a linear velocity field ``v(x) = scale * x``."""

    def __init__(self):
        super().__init__()
        from rlinf.models.embodiment.fastwam import flow_matching as fm

        self.action_scheduler = fm.FlowMatchingScheduler()
        # Components the policy expects to exist (freezing / device lookup);
        # their outputs are overridden by the encode_* stubs below.
        self.vae = nn.Linear(4, 4)
        self.text_encoder = nn.Linear(8, 8)
        self.proprio_encoder = nn.Linear(8, 8)
        self.video_expert = nn.Linear(4, 4)
        # Stands in for the "action expert" network being trained.
        self.scale = nn.Parameter(torch.tensor(-0.5))

    def encode_prompts(self, prompts):
        batch_size = len(prompts)
        seq_len = max(1, max(len(p) for p in prompts) // 5)
        return (
            torch.zeros(batch_size, seq_len, 8),
            torch.ones(batch_size, seq_len, dtype=torch.bool),
        )

    def encode_proprios(self, proprio):
        batch_size = proprio.shape[0]
        return (
            torch.zeros(batch_size, 1, 8),
            torch.ones(batch_size, 1, dtype=torch.bool),
        )

    def encode_videos(self, videos, device):
        return torch.zeros(videos.shape[0], 2, 1, 2, 2, device=device)

    def prepare_video_cache(
        self,
        first_frame_latents,
        action_horizon,
        prompt_embeds,
        prompt_embeds_mask=None,
    ):
        del first_frame_latents, action_horizon, prompt_embeds, prompt_embeds_mask
        return [], None

    def predict_action(
        self,
        action_latents,
        action_sigma,
        prompt_embeds,
        prompt_embeds_mask,
        video_kv_cache,
        attention_mask,
    ):
        del (
            action_sigma,
            prompt_embeds,
            prompt_embeds_mask,
            video_kv_cache,
            attention_mask,
        )
        return self.scale * action_latents


def _make_policy():
    pytest.importorskip("diffusers")
    pytest.importorskip("transformers")
    from rlinf.models.embodiment.fastwam import FastWAMActionPolicy

    policy = FastWAMActionPolicy(
        _FakeWAM(),
        action_min=-ones,
        action_max=ones,
        proprio_min=-ones,
        proprio_max=ones,
        num_flow_steps=4,
        noise_level=1.0,
        seed=5,
        cond_max_len=24,
    )
    policy.action_horizon = 8
    policy.action_dim = 7
    return policy


def _make_env_obs(batch_size=3, task_len=20):
    return {
        "main_images": torch.randint(
            0, 255, (batch_size, 32, 32, 3), dtype=torch.uint8
        ),
        "wrist_images": torch.randint(
            0, 255, (batch_size, 32, 32, 3), dtype=torch.uint8
        ),
        "states": torch.randn(batch_size, 7),
        "task_descriptions": ["t" * task_len] * batch_size,
    }


def _forward_inputs(result):
    return {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in result["forward_inputs"].items()
    }


def test_rollout_and_recompute_shapes():
    policy = _make_policy()
    actions, result = policy.predict_action_batch(_make_env_obs(), mode="train")
    assert actions.shape == (3, 8, 7)
    assert torch.isfinite(actions).all()
    log_probs = result["prev_logprobs"]
    assert log_probs.shape == (3, 8, 7)
    assert log_probs.dtype == torch.float32
    assert result["prev_values"].shape == (3, 1)
    assert set(result["forward_inputs"]) == {
        "x_t",
        "x_next",
        "denoise_inds",
        "first_frame_latents",
        "cond_embeds",
        "cond_mask",
    }

    actions_eval, result_eval = policy.predict_action_batch(
        _make_env_obs(), mode="eval"
    )
    assert actions_eval.shape == (3, 8, 7)
    assert result_eval["prev_logprobs"] is None


def test_recompute_matches_rollout_under_same_weights():
    """Flow-GRPO on-policy sanity: ratio == 1 when weights are unchanged."""
    policy = _make_policy()
    _, result = policy.predict_action_batch(_make_env_obs(), mode="train")
    output = policy.default_forward(_forward_inputs(result))
    torch.testing.assert_close(
        output["logprobs"], result["prev_logprobs"], rtol=0.0, atol=1e-5
    )
    assert output["values"].shape == (3, 1)
    assert output["entropy"].shape == (3,)


def test_gradient_reaches_action_expert():
    policy = _make_policy()
    _, result = policy.predict_action_batch(_make_env_obs(), mode="train")
    output = policy.default_forward(_forward_inputs(result))
    assert policy.model.scale.grad is None
    output["logprobs"].sum().backward()
    assert policy.model.scale.grad is not None
    assert torch.isfinite(policy.model.scale.grad).all()

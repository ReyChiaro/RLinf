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

"""Checkpoint-layout tests for FastWAM.

These tests do NOT instantiate the real (multi-billion-parameter) model. They
verify the loading logic that must hold regardless of the actual checkpoint:
the per-module config parsing (FastWAM-Trainer layout), the scheduler config
fallback, and the assembled model's state-dict key prefixes matching the
whole-model SFT snapshot prefixes (``vae`` / ``text_encoder`` /
``proprio_encoder`` / ``mot`` with experts under ``mot.mixtures.*``).
"""

import json

import pytest
import torch
import torch.nn as nn


def _load_modules():
    pytest.importorskip("diffusers")
    pytest.importorskip("transformers")
    from rlinf.models.embodiment.fastwam import checkpoint as ckpt
    from rlinf.models.embodiment.fastwam import flow_matching as fm
    from rlinf.models.embodiment.fastwam.llm import WanTextEncoder, WanTextEncoderConfig
    from rlinf.models.embodiment.fastwam.wam import FastWAM, ProprioEncoder

    return ckpt, fm, WanTextEncoder, WanTextEncoderConfig, FastWAM, ProprioEncoder


class _TinyExpert(nn.Module):
    """Cheap stand-in for a DiT expert (validates the key prefix contract)."""

    def __init__(self, name):
        super().__init__()
        self.register_parameter("weight", nn.Parameter(torch.zeros(4, 4)))
        self._name = name


def test_config_cleaning_skips_trainer_only_keys(tmp_path):
    ckpt, *_ = _load_modules()
    config = {
        "module": "action_expert",
        "action_dim": 7,
        "use_gradient_checkpointing": True,
        "hidden_dim": 128,
    }
    cleaned = ckpt._clean_component_config(config)
    assert cleaned == {"action_dim": 7, "hidden_dim": 128}


def test_single_scheduler_folder_parsed(tmp_path):
    ckpt, fm, *_ = _load_modules()
    scheduler_dir = tmp_path / "scheduler"
    scheduler_dir.mkdir()
    (scheduler_dir / "scheduler_config.json").write_text(
        json.dumps(
            {
                "video_scheduler": {"shift": 5.0, "num_training_steps": 1000},
                "action_scheduler": {"shift": 5.0, "num_training_steps": 1000},
            }
        )
    )
    video_scheduler, action_scheduler = ckpt._load_schedulers(tmp_path)
    assert isinstance(video_scheduler, fm.FlowMatchingScheduler)
    assert isinstance(action_scheduler, fm.FlowMatchingScheduler)
    assert video_scheduler.shift == action_scheduler.shift == 5.0


def test_assembled_state_dict_prefixes_match_sft_snapshot():
    """Prefix families of the assembled model equal the SFT snapshot's."""
    _ckpt, fm, WanTextEncoder, WanTextEncoderConfig, FastWAM, ProprioEncoder = (
        _load_modules()
    )

    vae = nn.Linear(4, 4)  # stand-in: only the 'vae' attribute name matters here
    video_dit = _TinyExpert("video")
    action_dit = _TinyExpert("action")
    proprio = ProprioEncoder(in_features=4, out_features=16)
    text_cfg = WanTextEncoderConfig(
        vocab=1024,
        dim=32,
        dim_attn=32,
        dim_ffn=64,
        num_heads=4,
        num_layers=1,
        num_buckets=8,
        shared_pos=False,
        dropout=0.0,
    )
    text_encoder = WanTextEncoder(text_cfg)

    model = FastWAM(
        vae=vae,
        tokenizer=object(),
        text_encoder=text_encoder,
        proprio_encoder=proprio,
        video_expert=video_dit,
        action_expert=action_dit,
        video_scheduler=fm.FlowMatchingScheduler(shift=5.0, num_training_steps=100),
        action_scheduler=fm.FlowMatchingScheduler(shift=5.0, num_training_steps=100),
    )

    prefixes = sorted({key.split(".")[0] for key in model.state_dict()})
    assert prefixes == ["mot", "proprio_encoder", "text_encoder", "vae"]
    keys = model.state_dict()
    assert any(k.startswith("mot.mixtures.video.") for k in keys)
    assert any(k.startswith("mot.mixtures.action.") for k in keys)
    assert any(k.startswith("vae.") for k in keys)

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

"""FastWAM embodied model support.

This package vendors the FastWAM world-action model definition (ported from
the FastWAM-RL project, Apache-2.0 — see the module headers of ``llm.py``,
``dit.py``, ``mot.py``, ``vae.py``, ``wam.py`` and ``flow_matching.py``) and
adds the RLinf-specific RL wrapper :class:`FastWAMActionPolicy` that plugs the
FastWAM action flow into RLinf's rollout/actor machinery (Flow-GRPO).
"""

from typing import Any

import torch
import torch.nn as nn

from rlinf.config import torch_dtype_from_precision

_MODEL_TYPE = "fastwam"


def get_model(cfg: Any, torch_dtype: Any = None) -> nn.Module:
    """Build the FastWAM policy from ``cfg`` (the ``actor.model`` DictConfig).

    Checkpoint layout (see ``rlinf.models.embodiment.fastwam.checkpoint``):

    * ``cfg.model_path`` points at a *per-module* checkpoint directory with one
      component subfolder: ``vae``, ``text_encoder``, ``video_expert``,
      ``action_expert``, ``proprio_encoder``, ``tokenizer``, plus a single
      ``scheduler/`` (nested ``video_scheduler``/``action_scheduler`` configs) or
      split ``video_scheduler/`` + ``action_scheduler/``. Weight files may use
      HuggingFace naming (``model.safetensors``, sharded ``model-XXXXX*.safetensors``
      + ``model.safetensors.index.json``) or vanilla diffusers naming
      (``diffusion_pytorch_model.*``).
    * ``cfg.sft_state_dict_path`` (optional) points at a whole-model SFT snapshot
      (``sft/weights`` sharded state dict, or one merged ``.safetensors``);
      when given it fully replaces the per-module weights after assembly.

    Normalization statistics are read from ``cfg.stats_path`` (the
    ``dataset_stats.json`` shipped with the dataset / weights; entries
    ``action`` and ``state``).

    Returns:
        The :class:`FastWAMActionPolicy` wrapper (frozen conditioning + the
        trainable action expert).
    """
    if torch_dtype is None:
        torch_dtype = torch_dtype_from_precision(cfg.get("precision", None))

    if cfg.get("is_lora", False):
        raise NotImplementedError(
            "FastWAM RL does not support LoRA: the Flow-GRPO objective updates "
            "the action expert only (set model.is_lora: False)."
        )

    from .checkpoint import build_fastwam_from_checkpoint
    from .policy import FastWAMActionPolicy

    model = build_fastwam_from_checkpoint(
        str(cfg.model_path),
        torch_dtype=torch_dtype,
        sft_state_dict_path=cfg.get("sft_state_dict_path", None),
    )

    action_min, action_max, proprio_min, proprio_max = _load_action_and_state_stats(cfg)

    policy = FastWAMActionPolicy(
        model,
        action_min=action_min,
        action_max=action_max,
        proprio_min=proprio_min,
        proprio_max=proprio_max,
        num_flow_steps=int(cfg.get("num_flow_steps", 10)),
        noise_level=float(cfg.get("noise_level", 1.0)),
        seed=cfg.get("seed", None),
        cond_max_len=int(cfg.get("cond_max_len", 512)),
    )
    policy.action_dim = int(cfg.get("action_dim", action_min.numel()))
    policy.action_horizon = int(cfg.get("num_action_chunks", 32))
    return policy


def _load_action_and_state_stats(cfg: Any):
    """Load min/max action & proprio statistics from ``cfg.stats_path``.

    The statistics file mirrors the ``dataset_stats.json`` produced with the
    LIBERO-fastwam SFT data: ``{"action": {"default": {"global_min": [...],
    "global_max": [...]}}, "state": {...}}``.
    """
    import json
    from pathlib import Path

    stats_path = cfg.get("stats_path", None)
    if stats_path is None:
        raise ValueError(
            "FastWAM requires model.stats_path pointing at the dataset_stats.json "
            "(action/state min-max used by the SFT checkpoint)."
        )
    with open(Path(stats_path), "r", encoding="utf-8") as f:
        raw = json.load(f)

    action_stats = raw.get("action", {})
    state_stats = raw.get("state", {})
    try:
        action_min = torch.as_tensor(
            action_stats["default"]["global_min"], dtype=torch.float32
        )
        action_max = torch.as_tensor(
            action_stats["default"]["global_max"], dtype=torch.float32
        )
        proprio_min = torch.as_tensor(
            state_stats["default"]["global_min"], dtype=torch.float32
        )
        proprio_max = torch.as_tensor(
            state_stats["default"]["global_max"], dtype=torch.float32
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"dataset_stats.json at {stats_path} must contain nested "
            "'action'/'state' -> 'default' -> 'global_min'/'global_max'."
        ) from exc
    return action_min, action_max, proprio_min, proprio_max


__all__ = ["FastWAMActionPolicy", "get_model"]


def __getattr__(name: str):
    """Lazy attribute access (PEP 562) to keep the heavy model import optional.

    ``get_model`` (and therefore the actual FastWAM/diffusers import) is only
    needed when the ``fastwam`` model type is used; direct ``from ...fastwam
    import FastWAMActionPolicy`` (tests, tooling) triggers it on demand.
    """
    if name == "FastWAMActionPolicy":
        from .policy import FastWAMActionPolicy

        return FastWAMActionPolicy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

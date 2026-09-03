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

"""Checkpoint loaders for the FastWAM model.

FastWAM-Trainer produces two kinds of artifacts that both need to be consumed:

1. **Per-module checkpoints** (e.g. ``FastWAM-Wan2.2-Init``): one diffusers-style
   subfolder per component (``vae``, ``text_encoder``, ``video_expert``,
   ``action_expert``, ``proprio_encoder``, ``tokenizer``) plus a single
   ``scheduler/`` holding the ``video_scheduler``/``action_scheduler`` configs.
   The weight files use HuggingFace naming (``model.safetensors`` or sharded
   ``model-XXXXX-of-YYYYY.safetensors`` + ``model.safetensors.index.json``),
   unlike vanilla diffusers ``diffusion_pytorch_model.*`` naming.

2. **Whole-model SFT snapshots** (``sft/weights``): a single sharded state dict
   of the assembled :class:`FastWAM` whose top-level key prefixes are ``vae``,
   ``text_encoder``, ``proprio_encoder`` and ``mot`` (the two experts live at
   ``mot.mixtures.video.*`` / ``mot.mixtures.action.*``). Applying it is
   equivalent to a full fine-tune on top of the per-module checkpoint.

:func:`build_fastwam_from_checkpoint` assembles the model from the per-module
configs, then loads either the per-module weights or (preferred) the SFT
whole-model state dict.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional, Union

import torch

logger = logging.getLogger(__name__)

_COMPONENT_SUBFOLDERS = (
    "vae",
    "text_encoder",
    "video_expert",
    "action_expert",
    "proprio_encoder",
    "tokenizer",
)

# Config keys that are part of the trainer manifests but not constructor args.
_SKIP_CONFIG_KEYS = ("module", "use_gradient_checkpointing")


def _read_json(path: Union[str, Path]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _clean_component_config(config: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in config.items() if k not in _SKIP_CONFIG_KEYS}


def _resolve_weight_files(component_dir: Path) -> list[Path]:
    """Weight files of one component dir (HF *or* diffusers naming, incl. shards).

    Returns an ordered list of ``.safetensors`` files (all shards when sharded,
    otherwise the single weight file).
    """
    files = sorted(component_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(
            f"No .safetensors weights in {component_dir}; expected "
            "'model.safetensors' / 'diffusion_pytorch_model.safetensors' or "
            "sharded 'model-*.safetensors' / 'diffusion_pytorch_model-*.safetensors'."
        )
    index = sorted(component_dir.glob("*.index.json"))
    if index:
        # Sharded: keep the files the index references, in shard order.
        referenced = set(_read_json(index[0])["weight_map"].values())
        return [p for p in files if p.name in referenced]
    # A single-file layout may still have left-over shards named like the
    # index-less single file only; keep everything (sorted) otherwise.
    return files


def _load_state_dict_from_dir(component_dir: Path) -> dict[str, torch.Tensor]:
    """Load and merge (possibly sharded) safetensors of one component dir."""
    from safetensors.torch import load_file

    state_dict: dict[str, torch.Tensor] = {}
    for shard in _resolve_weight_files(component_dir):
        state_dict.update(load_file(str(shard)))
    return state_dict


def _load_module_weights(
    module: torch.nn.Module,
    component_dir: Path,
    torch_dtype: Optional[torch.dtype],
) -> None:
    """Load one component's weight file(s) into ``module`` (in-place)."""
    state_dict = _load_state_dict_from_dir(component_dir)
    if torch_dtype is not None:
        state_dict = {
            key: value.to(torch_dtype) if value.is_floating_point() else value
            for key, value in state_dict.items()
        }
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Loading {component_dir} into {type(module).__name__} failed: "
            f"missing={missing[:8]}... unexpected={unexpected[:8]}..."
        )


def _instantiate_text_encoder(config_path: Path) -> torch.nn.Module:
    """Build the text encoder (transformers model) from its own config.json."""
    from .llm import WanTextEncoder, WanTextEncoderConfig

    config = WanTextEncoderConfig(**_clean_component_config(_read_json(config_path)))
    return WanTextEncoder(config)


def _load_tokenizer(
    root: Path,
    seq_len: Optional[int],
    clean: Optional[str],
) -> Any:
    """Build the prompt tokenizer from the HF files of ``root/tokenizer``.

    FastWAM-Trainer checkpoints store only the standard HuggingFace files in
    the tokenizer folder (``tokenizer_config.json`` / ``spiece.model`` / …);
    the ``seq_len`` / ``clean`` companion config is carried in
    ``root/model_config.yaml`` instead of a ``wan_text_encoder_tokenizer_config.json``.
    """
    from .llm import WantTextEncoderTokenizer

    if seq_len is None or clean is None:
        try:
            import yaml
        except ImportError:  # pragma: no cover
            yaml = None
        tok_cfg: dict[str, Any] = {}
        model_config_path = root / "model_config.yaml"
        if yaml is not None and model_config_path.is_file():
            with open(model_config_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            if isinstance(loaded, dict) and isinstance(loaded.get("tokenizer"), dict):
                tok_cfg = loaded["tokenizer"]
        if seq_len is None:
            seq_len = int(tok_cfg.get("seq_len", 128))
        if clean is None:
            clean = str(tok_cfg.get("clean", "whitespace"))
    return WantTextEncoderTokenizer(
        str(root / "tokenizer"), seq_len=seq_len, clean=clean
    )


def _instantiate(config_path: Path, cls: type) -> torch.nn.Module:
    """Instantiate a torch component from its ``config.json`` (in-place)."""
    config = _clean_component_config(_read_json(config_path))
    module = cls(**config)
    raw = _read_json(config_path)
    if raw.get("use_gradient_checkpointing", False):
        module.use_gradient_checkpointing = True
    return module


def _load_schedulers(root: Path):
    """Build the video/action flow schedulers.

    Supports the FastWAM-Trainer single ``scheduler/`` folder (whose
    ``scheduler_config.json`` nests ``video_scheduler``/``action_scheduler``)
    and the upstream split ``video_scheduler/`` + ``action_scheduler/`` layout.
    """
    from .flow_matching import FlowMatchingScheduler

    scheduler_dir = root / "scheduler"
    if scheduler_dir.is_dir():
        cfg = _read_json(scheduler_dir / "scheduler_config.json")
        video_cfg = cfg.get("video_scheduler")
        action_cfg = cfg.get("action_scheduler")
        if video_cfg is None or action_cfg is None:
            raise ValueError(
                "scheduler/scheduler_config.json must contain both "
                "'video_scheduler' and 'action_scheduler' configs."
            )
        video_kwargs = {
            k: v
            for k, v in video_cfg.items()
            if not str(k).startswith("_target_") and k != "_target_"
        }
        action_kwargs = {
            k: v
            for k, v in action_cfg.items()
            if not str(k).startswith("_target_") and k != "_target_"
        }
        return (
            FlowMatchingScheduler(**video_kwargs),
            FlowMatchingScheduler(**action_kwargs),
        )

    def _scheduler(subfolder: str) -> FlowMatchingScheduler:
        sub_dir = root / subfolder
        if not (sub_dir / "scheduler_config.json").is_file():
            raise FileNotFoundError(
                f"Expected {sub_dir / 'scheduler_config.json'} (split scheduler "
                "layout) or a single scheduler/ folder with nested configs."
            )
        return FlowMatchingScheduler.from_pretrained(str(root), subfolder=subfolder)

    return _scheduler("video_scheduler"), _scheduler("action_scheduler")


def build_fastwam_from_checkpoint(
    checkpoint_dir: Union[str, Path],
    *,
    torch_dtype: Optional[torch.dtype] = None,
    sft_state_dict_path: Optional[Union[str, Path]] = None,
    tokenizer_seq_len: Optional[int] = None,
    tokenizer_clean: Optional[str] = None,
) -> torch.nn.Module:
    """Assemble the FastWAM model from a checkpoint directory (see module doc).

    Args:
        checkpoint_dir: Root of the per-module checkpoint (init weights).
        torch_dtype: Cast floating-point weights to this dtype.
        sft_state_dict_path: Optional whole-model SFT snapshot (a directory with
            sharded ``model.safetensors`` + index, or an already merged
            ``.safetensors`` file). When given, its state dict fully replaces
            the per-module weights (SFT trained from the init checkpoint).
        tokenizer_seq_len / tokenizer_clean: Explicit prompt-tokenizer settings.
            When omitted they fall back to ``root/model_config.yaml``
            (``tokenizer.seq_len`` / ``tokenizer.clean``) and finally to
            ``128`` / ``"whitespace"``.
    """
    from .dit import ActionDiT, VideoDiT
    from .vae import WanVideoVAE
    from .wam import FastWAM, ProprioEncoder

    root = Path(checkpoint_dir)
    missing = [sub for sub in _COMPONENT_SUBFOLDERS if not (root / sub).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"FastWAM checkpoint {root} misses subfolders: {missing}. Expected: "
            f"{list(_COMPONENT_SUBFOLDERS)}."
        )

    vae = _instantiate(root / "vae" / "config.json", WanVideoVAE)
    video_dit = _instantiate(root / "video_expert" / "config.json", VideoDiT)
    action_dit = _instantiate(root / "action_expert" / "config.json", ActionDiT)
    proprio = _instantiate(root / "proprio_encoder" / "config.json", ProprioEncoder)
    text_encoder = _instantiate_text_encoder(root / "text_encoder" / "config.json")
    tokenizer = _load_tokenizer(root, tokenizer_seq_len, tokenizer_clean)
    video_scheduler, action_scheduler = _load_schedulers(root)

    model = FastWAM(
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        proprio_encoder=proprio,
        video_expert=video_dit,
        action_expert=action_dit,
        video_scheduler=video_scheduler,
        action_scheduler=action_scheduler,
    )

    if sft_state_dict_path is not None:
        sft_dir = Path(sft_state_dict_path)
        if sft_dir.is_dir():
            state_dict = _load_state_dict_from_dir(sft_dir)
        else:
            from safetensors.torch import load_file

            state_dict = load_file(str(sft_dir))
        if torch_dtype is not None:
            state_dict = {
                key: value.to(torch_dtype) if value.is_floating_point() else value
                for key, value in state_dict.items()
            }
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                "SFT whole-model state dict does not match the assembled FastWAM: "
                f"missing={missing_keys[:8]}... unexpected={unexpected_keys[:8]}..."
            )
    else:
        # Load each component from its own per-module weights.
        module_targets = (
            (model.vae, root / "vae"),
            (model.text_encoder, root / "text_encoder"),
            (model.video_expert, root / "video_expert"),
            (model.action_expert, root / "action_expert"),
            (model.proprio_encoder, root / "proprio_encoder"),
        )
        for module, component_dir in module_targets:
            _load_module_weights(module, component_dir, torch_dtype)

    if torch_dtype is not None:
        model = model.to(torch_dtype)
    return model

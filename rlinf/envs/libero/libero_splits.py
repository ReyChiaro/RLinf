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

"""Deterministic train/test task splits for the LIBERO-Plus benchmark.

LIBERO-Plus (a drop-in ``libero`` fork) expands each of the four vanilla task
suites into ~10,000 perturbed tasks built from exactly 10 base stems per suite
(e.g. ``pick_up_the_black_bowl_...`` + perturbation suffixes such as
``_view_0_0_100_0_0_initstate_17``, ``_language_3``, ``_noise_12``...).

A meaningful RL split must not leak a base skill across train and test, so we
group full task ids by their *base stem* (perturbation suffixes stripped) and
split on stem groups with a fixed seed. Task ids resolve to integer indices in
the installed benchmark's task order, which is what RLinf's ``LiberoEnv``
``task_id_filter`` consumes.

Usage from an env config (see ``examples/embodiment/config/env/``):

.. code-block:: yaml

   task_suite_name: libero_spatial
   suite_split:
     seed: 0
     subset: train        # "train" or "test"
     train_fraction: 0.8  # fraction of base stems used for training

Run ``python -m rlinf.envs.libero.libero_splits --help`` for the CLI that
prints the resolved task ids / indices of a suite subset.
"""

from __future__ import annotations

import argparse
import logging
import re
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Perturbation suffixes of LIBERO-Plus task ids, ordered so that nested
# suffixes (e.g. ``_view_..._initstate_0_noise_5``) are stripped from the
# outermost to the innermost part by repeated application.
_PERTURBATION_SUFFIX_PATTERNS = [
    # camera viewpoint + optional sensor noise, e.g. stem_view_0_0_100_0_0_initstate_3_noise_7
    re.compile(r"_noise_\d+$"),
    re.compile(r"_view_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_initstate_(\d+)$"),
    # robot initial states, e.g. stem_view_0_0_100_0_0_initstate_3 (no view tuple
    # prefix) is expressed via the following catch-all:
    re.compile(r"_initstate_(\d+)$"),
    # background textures
    re.compile(r"_table_(\d+)$"),
    re.compile(r"_tb_(\d+)$"),
    # language rewrites
    re.compile(r"_language_(\d+)$"),
    # light conditions
    re.compile(r"_light_(\d+)$"),
    # object layout variants
    re.compile(r"_add_(\d+)$"),
    re.compile(r"_moved_level(\d+)_sample(\d+)$"),
    re.compile(r"_level(\d+)_sample(\d+)$"),
]

_SUPPORTED_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def strip_perturbation_suffixes(task_id: str) -> str:
    """Return the base task stem of a (possibly perturbed) LIBERO-Plus task id."""
    stem = task_id
    while True:
        next_stem = stem
        for pattern in _PERTURBATION_SUFFIX_PATTERNS:
            next_stem = pattern.sub("", next_stem)
        if next_stem == stem:
            return stem
        stem = next_stem


def split_stems_by_seed(
    task_ids: list[str],
    *,
    seed: int = 0,
    train_fraction: float = 0.8,
) -> tuple[set[str], set[str]]:
    """Split base stems deterministically into train/test stem sets."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")
    stems = sorted({strip_perturbation_suffixes(tid) for tid in task_ids})
    if not stems:
        raise ValueError("cannot split an empty task list")
    rng = np.random.RandomState(seed)
    order = list(range(len(stems)))
    rng.shuffle(order)
    n_train = max(1, int(round(len(stems) * train_fraction)))
    if n_train >= len(stems):
        n_train = len(stems) - 1
    train_stems = {stems[i] for i in order[:n_train]}
    test_stems = {stems[i] for i in order[n_train:]}
    return train_stems, test_stems


def suite_task_ids(suite_name: str) -> list[str]:
    """Full ordered task-id list of a suite from the installed ``libero`` package.

    Prefers the LIBERO-Plus flat registry
    (``libero.libero.benchmark.libero_suite_task_map.libero_task_map``) and
    falls back to the benchmark task objects exposed by the installed fork /
    vanilla package through RLinf's benchmark resolution.
    """
    suite_name = str(suite_name).lower()
    if suite_name not in _SUPPORTED_SUITES:
        raise NotImplementedError(
            f"suite_split supports {_SUPPORTED_SUITES}; got {suite_name}. "
            "Add the suite to rlinf.envs.libero.libero_splits if needed."
        )

    # LIBERO-Plus ships one python module with every full task id, in file
    # order == benchmark index order.
    try:
        from libero.libero.benchmark import (
            libero_suite_task_map as _task_map_mod,  # type: ignore
        )

        task_map = getattr(_task_map_mod, "libero_task_map", None)
        ids = task_map.get(suite_name) if task_map else None
        if ids:
            return list(ids)
    except ImportError:  # pragma: no cover - depends on user installation
        pass

    # Fallback: whatever ``libero`` package is installed (fork or vanilla).
    try:
        from rlinf.envs.libero.utils import get_benchmark_overridden

        benchmark = get_benchmark_overridden(suite_name)()
        return [task.name for task in benchmark.tasks]
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            f"Could not resolve LIBERO-Plus task ids for suite {suite_name!r}. "
            "Install the LIBERO-Plus fork (pip install -e . from the "
            "LIBERO-plus repo) or make the vanilla libero package importable."
        ) from exc


def suite_split_task_indices(
    suite_name: str,
    *,
    subset: str = "train",
    seed: int = 0,
    train_fraction: float = 0.8,
) -> list[int]:
    """Task indices (benchmark order) of the train/test subset of a suite.

    The split is computed on base stems so perturbed variants of one origin
    task never appear on both sides of the split.
    """
    if subset not in ("train", "test"):
        raise ValueError(f"subset must be 'train' or 'test', got {subset!r}")
    task_ids = suite_task_ids(suite_name)
    train_stems, test_stems = split_stems_by_seed(
        task_ids, seed=seed, train_fraction=train_fraction
    )
    wanted = train_stems if subset == "train" else test_stems
    return [
        index
        for index, task_id in enumerate(task_ids)
        if strip_perturbation_suffixes(task_id) in wanted
    ]


def resolve_suite_split(
    suite_name: str,
    cfg: Any,
) -> Optional[list[int]]:
    """Resolve ``cfg.suite_split`` (dict / OmegaConf) into a task-id-filter list.

    Returns ``None`` when no ``suite_split`` block is configured (i.e. the
    whole suite is used, RLinf's default behavior).
    """
    split_cfg = cfg.get("suite_split") if hasattr(cfg, "get") else None
    if not split_cfg:
        return None
    return suite_split_task_indices(
        suite_name,
        subset=str(split_cfg.get("subset", "train")),
        seed=int(split_cfg.get("seed", 0)),
        train_fraction=float(split_cfg.get("train_fraction", 0.8)),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print train/test task ids of a LIBERO-Plus suite.",
    )
    parser.add_argument("--suite", required=True, choices=_SUPPORTED_SUITES)
    parser.add_argument("--subset", choices=["train", "test"], default="train")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--no-index", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    indices = suite_split_task_indices(
        args.suite,
        subset=args.subset,
        seed=args.seed,
        train_fraction=args.train_fraction,
    )
    task_ids = suite_task_ids(args.suite)
    lines = [task_ids[i] if not args.no_index else str(i) for i in indices]
    print(f"# suite={args.suite} subset={args.subset} seed={args.seed} n={len(lines)}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

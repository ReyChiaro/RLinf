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

"""Tests for the LIBERO-Plus deterministic train/test task splits."""

import pytest

from rlinf.envs.libero.libero_splits import (
    split_stems_by_seed,
    strip_perturbation_suffixes,
    suite_split_task_indices,
)


def test_strip_perturbation_suffixes():
    cases = {
        "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate": (
            "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate"
        ),
        "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_3": (
            "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate"
        ),
        "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_table_12": (
            "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate"
        ),
        "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_language_4_view_359_15_100_0_0_initstate_0": (
            "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate"
        ),
        "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_light_7": (
            "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate"
        ),
        "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_add_21": (
            "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate"
        ),
        "KITCHEN_SCENE3_turn_on_the_stove_view_0_0_100_0_0_initstate_5_noise_17": (
            "KITCHEN_SCENE3_turn_on_the_stove"
        ),
        "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_moved_level3_sample2": (
            "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket"
        ),
    }
    for task_id, expected in cases.items():
        assert strip_perturbation_suffixes(task_id) == expected


def _fake_suite_ids():
    # Two base stems x several perturbation variants.
    return [
        f"task_a_{suffix}"
        for suffix in ["view_0_0_100_0_0_initstate_1", "table_3", "light_5", "noise_9"]
    ] + [
        f"task_b_{suffix}"
        for suffix in ["view_1_1_100_0_0_initstate_2", "add_4", "language_2"]
    ]


def test_split_stems_are_disjoint_and_seeded():
    ids = _fake_suite_ids()
    train_a, test_a = split_stems_by_seed(ids, seed=0, train_fraction=0.5)
    train_b, test_b = split_stems_by_seed(ids, seed=0, train_fraction=0.5)
    assert train_a == train_b
    assert test_a == test_b
    assert train_a.isdisjoint(test_a)
    assert train_a | test_a == {"task_a", "task_b"}


def test_suite_split_task_indices_consistent(monkeypatch):
    from rlinf.envs.libero import libero_splits

    ids = _fake_suite_ids()
    monkeypatch.setattr(libero_splits, "suite_task_ids", lambda suite: ids)

    train = suite_split_task_indices("libero_spatial", subset="train", seed=0)
    test = suite_split_task_indices("libero_spatial", subset="test", seed=0)
    assert sorted(train + test) == sorted(range(len(ids)))
    assert train and test
    # No base stem appears on both sides.
    stems = {strip_perturbation_suffixes(ids[i]) for i in set(train) & set(test)}
    assert not stems


def test_resolve_suite_split_disabled():
    from rlinf.envs.libero.libero_splits import resolve_suite_split

    assert resolve_suite_split("libero_spatial", {}) is None
    with pytest.raises(ValueError):
        resolve_suite_split("libero_spatial", {"suite_split": {"subset": "bad"}})

import pytest
import torch

from tools.export_vlm_actionness_candidates import choose_indices, named_paths
from tools.sample_vlm_hard_case_challenge import balanced_sample, classify, interleave


def test_choose_indices_supports_low_and_high_score_bands():
    scores = torch.tensor([0.05, 0.2, 0.49, 0.5, 0.9])
    centers = torch.arange(5, dtype=torch.float32)
    valid = torch.ones(5, dtype=torch.bool)

    low = choose_indices(scores, centers, valid, 0.1, 0.5, 10, 0, "low")
    high = choose_indices(scores, centers, valid, 0.1, 0.5, 10, 0, "high")

    assert low == [2, 1]
    assert high == [4, 3]


def test_choose_indices_rejects_unknown_score_band():
    with pytest.raises(ValueError, match="score_band"):
        choose_indices(
            torch.tensor([0.5]),
            torch.tensor([0.0]),
            torch.tensor([True]),
            0.1,
            0.5,
            1,
            0,
            "all",
        )


def test_named_validity_models_are_unique_and_explicit():
    assert named_paths(["a=/tmp/a.json", "b=/tmp/b.json"]) == {
        "a": "/tmp/a.json",
        "b": "/tmp/b.json",
    }
    with pytest.raises(ValueError, match="NAME=PATH"):
        named_paths(["missing-name-separator"])
    with pytest.raises(ValueError, match="duplicate"):
        named_paths(["a=/tmp/a.json", "a=/tmp/b.json"])


def test_hard_case_groups_have_distinct_semantics():
    base = {
        "gt_is_foreground": False,
        "center_in_target_action": False,
        "sada_top1_is_correct": False,
    }
    assert classify({**base, "score_band": "low"}) == "low_background"
    assert classify({**base, "score_band": "high"}) == "high_background"
    assert classify(
        {**base, "score_band": "low", "center_in_target_action": True}
    ) == "low_semantic_invalid"
    assert classify(
        {
            **base,
            "score_band": "low",
            "gt_is_foreground": True,
            "sada_top1_is_correct": True,
        }
    ) == "low_valid_correct"


def test_balanced_sample_and_interleave_do_not_start_from_one_cell_or_group():
    rows = [
        {
            "video_id": video,
            "fpn_level": level,
            "absolute_feature_index": level * 10 + value,
            "value": value,
        }
        for video in ("a", "b")
        for level in (0, 1)
        for value in range(3)
    ]
    sampled = balanced_sample(rows, 4, seed=7)
    assert len({(row["video_id"], row["fpn_level"]) for row in sampled}) == 4

    duplicate_time = [
        {"video_id": "a", "fpn_level": level, "absolute_feature_index": 8}
        for level in (0, 1)
    ]
    assert len(balanced_sample(duplicate_time, 2, seed=7)) == 1

    groups = {name: [{"group": name}] for name in (
        "low_valid_correct",
        "low_semantic_invalid",
        "low_background",
        "high_background",
        "high_correct_control",
    )}
    assert [row["group"] for row in interleave(groups)] == list(groups)

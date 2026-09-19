import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from tools.run_rq2_temporal_probe import (
    build_donor_indices,
    decode_probability_batches,
    interval_metrics,
    membership_role,
    nonvisual_fold_scores,
    prepare_feature_bank,
    ProbeRuntime,
    role_indices,
)


def row(candidate_id, instance_id, video_id, position, role="source_train"):
    start = 10.0
    end = 12.0
    return {
        "candidate_id": candidate_id,
        "instance_id": instance_id,
        "video_id": video_id,
        "context_position": position,
        "gt_label_id": 0,
        "gt_start_sec": start,
        "gt_end_sec": end,
        "proposal_start_sec": 8.0,
        "proposal_end_sec": 16.0,
        "window_length_sec": 8.0,
        "memberships": [{"shift": "S1", "role": role}],
    }


def test_role_indices_are_shift_and_role_specific():
    rows = [
        row("a", "i1", "v1", 0.25, "source_train"),
        row("b", "i2", "v2", 0.50, "source_val"),
        row("c", "i3", "v3", 0.75, "target_val"),
    ]
    assert role_indices(rows, "S1", "source_train").tolist() == [0]
    assert role_indices(rows, "S1", "source_val").tolist() == [1]
    assert role_indices(rows, "S1", "target_val").tolist() == [2]
    assert membership_role(rows[0], "S2") is None


def test_duplicate_roles_for_one_shift_are_rejected():
    item = row("a", "i1", "v1", 0.25)
    item["memberships"].append({"shift": "S1", "role": "target_val"})
    with pytest.raises(ValueError, match="multiple roles"):
        membership_role(item, "S1")


def test_decode_enforces_start_not_after_end():
    start = np.asarray([[0.1, 0.2, 0.7]], dtype=np.float32)
    end = np.asarray([[0.8, 0.1, 0.1]], dtype=np.float32)
    starts, ends = decode_probability_batches(start, end)
    assert starts.tolist() == [0]
    assert ends.tolist() == [0]


def test_shuffle_donors_preserve_class_and_change_video():
    indices = np.arange(6, dtype=np.int64)
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    videos = np.asarray(["a", "b", "c", "d", "e", "f"])
    donor = build_donor_indices(indices, labels, videos, seed=7)
    assert set(donor) == set(indices.tolist())
    assert set(donor.values()).issubset(set(indices.tolist()))
    for source, target in donor.items():
        assert labels[source] == labels[target]
        assert videos[source] != videos[target]


def test_shuffle_donors_match_temporal_length():
    indices = np.arange(8, dtype=np.int64)
    labels = np.zeros(8, dtype=np.int64)
    videos = np.asarray(["a", "b", "c", "d", "e", "f", "g", "h"])
    lengths = np.asarray([2, 2, 2, 2, 4, 4, 4, 4])
    donor = build_donor_indices(indices, labels, videos, seed=7, strata=lengths)
    for source, target in donor.items():
        assert lengths[source] == lengths[target]
        assert videos[source] != videos[target]


def test_shuffle_donors_fall_back_across_lengths_but_not_videos():
    indices = np.arange(4, dtype=np.int64)
    labels = np.zeros(4, dtype=np.int64)
    videos = np.asarray(["a", "a", "b", "b"])
    lengths = np.asarray([2, 2, 4, 4])
    donor = build_donor_indices(indices, labels, videos, seed=7, strata=lengths)
    for source, target in donor.items():
        assert videos[source] != videos[target]
        assert lengths[source] != lengths[target]


def test_context_metrics_require_all_three_predictions_to_be_correct():
    rows = [
        row("a", "same", "v1", 0.25),
        row("b", "same", "v1", 0.50),
        row("c", "same", "v1", 0.75),
    ]
    timestamps = np.asarray(
        [
            [9.0, 10.0, 11.0, 12.0],
            [9.0, 10.0, 11.0, 12.0],
            [9.0, 10.0, 11.0, 12.0],
        ],
        dtype=np.float32,
    )
    start = np.asarray([[0.0, 1.0, 0.0, 0.0]] * 3, dtype=np.float32)
    end = np.asarray([[0.0, 0.0, 0.0, 1.0]] * 3, dtype=np.float32)
    lengths = np.asarray([4, 4, 4], dtype=np.int64)
    metrics = interval_metrics(rows, np.arange(3), timestamps, lengths, start, end)
    assert metrics["miou"] == pytest.approx(1.0)
    assert metrics["context_consistency"] == pytest.approx(1.0)
    assert metrics["consistent_and_correct_at_0.5"] == pytest.approx(1.0)


def test_nonvisual_prior_uses_only_source_fitted_relative_statistics():
    rows = [
        row("a", "i1", "v1", 0.25),
        row("b", "i2", "v2", 0.50),
    ]
    scores = nonvisual_fold_scores(
        rows, np.asarray([0]), np.asarray([1]), seed=3
    )
    assert scores["center_duration"] == pytest.approx(1.0)
    assert scores["full_window"] == pytest.approx(0.25)


def test_feature_bank_supports_variable_time_lengths(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest_rows = [
        {"candidate_id": "a", "frame_count": 2},
        {"candidate_id": "b", "frame_count": 4},
    ]
    manifest.write_text(
        "".join(json.dumps(item) + "\n" for item in manifest_rows),
        encoding="utf-8",
    )
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    feature_root = tmp_path / "features"
    feature_root.mkdir()
    np.savez(
        feature_root / "a.npz",
        features=np.ones((2, 3), dtype=np.float32),
        timestamps_sec=np.asarray([0.0, 0.5], dtype=np.float32),
    )
    np.savez(
        feature_root / "b.npz",
        features=np.ones((4, 3), dtype=np.float32),
        timestamps_sec=np.asarray([1.0, 1.5, 2.0, 2.5], dtype=np.float32),
    )
    index = tmp_path / "index.jsonl"
    index_rows = [
        {
            "candidate_id": candidate_id,
            "status": "ok",
            "schema_version": "rq2-qwen-time-aligned-vision-tower-v1",
            "model": "test-model",
            "model_revision": "test-revision",
            "manifest_sha256": manifest_hash,
            "feature_file": candidate_id + ".npz",
        }
        for candidate_id in ("a", "b")
    ]
    index.write_text(
        "".join(json.dumps(item) + "\n" for item in index_rows),
        encoding="utf-8",
    )

    features, timestamps, lengths, offsets, metadata = prepare_feature_bank(
        manifest_rows,
        manifest,
        [index],
        feature_root,
        tmp_path / "cache",
    )

    assert features.shape == (6, 3)
    assert timestamps.shape == (2, 4)
    assert lengths.tolist() == [2, 4]
    assert offsets.tolist() == [0, 2, 6]
    assert np.isnan(timestamps[0, 2:]).all()
    assert metadata["feature_shape"] == [6, 3]


def test_probe_masks_padded_time_steps():
    rows = [
        row("a", "i1", "v1", 0.25),
        row("b", "i2", "v2", 0.50),
    ]
    features = np.arange(18, dtype=np.float32).reshape(6, 3) + 1.0
    timestamps = np.asarray(
        [[0.0, 0.5, np.nan, np.nan], [0.0, 0.5, 1.0, 1.5]],
        dtype=np.float32,
    )
    lengths = np.asarray([2, 4], dtype=np.int64)
    offsets = np.asarray([0, 2, 6], dtype=np.int64)
    args = SimpleNamespace(device="cpu", batch_size=2, eval_batch_size=2)
    runtime = ProbeRuntime(features, timestamps, lengths, offsets, rows, args)
    model = runtime.new_model(seed=0)

    start, end = runtime.probabilities(model, np.asarray([0, 1]))

    assert start[0, 2:].tolist() == [0.0, 0.0]
    assert end[0, 2:].tolist() == [0.0, 0.0]
    assert start.sum(axis=1).tolist() == pytest.approx([1.0, 1.0])
    assert end.sum(axis=1).tolist() == pytest.approx([1.0, 1.0])

import json
from types import SimpleNamespace

import numpy as np
from PIL import Image

from tools.build_rq1_semantic_manifest import midpoint_labels
from tools.evaluate_rq1_semantics import (
    aggregate_shifts,
    confusion_metrics,
    jensen_shannon,
)
from tools.build_rq1_semantic_manifest import SHIFT_CONFIGS
from tools.run_rq1_tuned_feature_probe import (
    grouped_cv_splits,
    select_alpha_one_standard_error,
)
from tools.audit_rq1_feature_protocol import audit_splits
from tools.merge_and_shard_rq1 import merge_rows, write_jsonl_atomic
from tools import prepare_rq1_frames
from tools.run_rq1_slowfast_probe import segment_feature
from tools.rq1_feature_store import FeatureStore, normalize, summarize_frame_features
from tools.extract_rq1_qwen_features import select_shard
from tools.extract_rq1_videochat3_features import video_metadata
from tools.run_rq1_frozen_feature_probe import source_train_indices


def test_midpoint_labels_marks_overlapping_top10_actions():
    annotations = [
        {"label_id": 0, "segment": [1.0, 3.0]},
        {"label_id": 1, "segment": [1.5, 2.5]},
        {"label_id": 15, "segment": [1.0, 3.0]},
    ]
    assert midpoint_labels(annotations, 0) == [0, 1]


def test_confusion_metrics_uses_macro_classes():
    confusion = np.zeros((10, 10), dtype=np.int64)
    confusion[0, 0] = 2
    confusion[1, 0] = 1
    metrics = confusion_metrics(confusion)
    assert metrics["accuracy"] == 2 / 3
    assert metrics["macro_recall"] == 0.1


def test_jensen_shannon_is_zero_for_equal_distributions():
    assert jensen_shannon([0.2, 0.8], [0.2, 0.8]) == 0.0


def test_segment_feature_pools_feature_grids_inside_interval():
    features = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    pooled = segment_feature(features, [0.0, 1.0], fps=1.0, feat_stride=1.0, num_frames=0.0)
    assert np.allclose(pooled, np.asarray([1.0, 1.0]) / np.sqrt(2.0))


def test_merge_rows_keeps_only_latest_successful_prediction(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    write_jsonl_atomic(first, [
        {"candidate_id": "a", "status": "ok", "value": 1},
        {"candidate_id": "b", "status": "error"},
    ])
    write_jsonl_atomic(second, [
        {"candidate_id": "a", "status": "ok", "value": 2},
        {"candidate_id": "b", "status": "ok", "value": 3},
    ])
    merged = merge_rows([first, second])
    assert merged["a"]["value"] == 2
    assert merged["b"]["value"] == 3


def test_prepare_frames_video_subset_uses_selected_segment_count(tmp_path, monkeypatch):
    rows = [
        {"video_id": "video_a", "candidate_id": "a"},
        {"video_id": "video_b", "candidate_id": "b"},
    ]
    args = SimpleNamespace(
        manifest="manifest.jsonl",
        video_root="videos",
        frame_cache=str(tmp_path),
        cache_namespace="test",
        frame_count=8,
        workers=1,
        video_id=["video_a"],
        force=True,
    )
    monkeypatch.setattr(prepare_rq1_frames, "parse_args", lambda: args)
    monkeypatch.setattr(prepare_rq1_frames, "read_jsonl", lambda _: rows)
    monkeypatch.setattr(
        prepare_rq1_frames,
        "prepare_video",
        lambda video_id, video_rows, *_: (video_id, len(video_rows), None),
    )

    assert prepare_rq1_frames.main() == 0
    report = json.loads((tmp_path / "test_prepare_report.json").read_text(encoding="utf-8"))
    assert report["total_segments"] == 1
    assert report["prepared_segments"] == 1


def test_frames_complete_rejects_a_truncated_jpeg(tmp_path):
    output = tmp_path / "frames"
    output.mkdir()
    path = prepare_rq1_frames.frame_path(output, 1)
    Image.new("RGB", (8, 8), "white").save(path)
    payload = path.read_bytes()
    path.write_bytes(payload[:-8])
    assert not prepare_rq1_frames.frames_complete(output, 1)


def test_prepare_video_rejects_an_active_aria_download(tmp_path):
    video = tmp_path / "P01_01.MP4"
    video.write_bytes(b"partial")
    (tmp_path / "P01_01.MP4.aria2").write_bytes(b"progress")
    video_id, count, error = prepare_rq1_frames.prepare_video(
        "P01_01", [], tmp_path, tmp_path / "frames", 8
    )
    assert (video_id, count, error) == ("P01_01", 0, "video missing")


def test_source_train_indices_excludes_validation_memberships():
    rows = [
        {
            "memberships": [
                {"shift": "S1", "role": "source_train"},
                {"shift": "S2", "role": "source_val"},
            ]
        },
        {"memberships": [{"shift": "S1", "role": "target_val"}]},
    ]
    assert source_train_indices(rows, "S1") == [0]
    assert source_train_indices(rows, "S2") == []


def test_frozen_feature_pooling_is_normalized_and_temporal():
    frames = np.asarray([
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ])
    static = summarize_frame_features(frames, "static")
    combined = summarize_frame_features(frames, "static_transition")
    assert np.allclose(static, normalize([1.0, 1.0]))
    assert combined.shape == (4,)
    assert combined[2] < 0.0 < combined[3]


def test_feature_store_resumes_completed_rows(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"candidate_id":"a"}\n{"candidate_id":"b"}\n', encoding="utf-8")
    rows = [{"candidate_id": "a"}, {"candidate_id": "b"}]
    prefix = tmp_path / "features" / "clip"
    store = FeatureStore(prefix, manifest, rows, "clip", 2, 8, "static")
    store.write([0], [[1.0, 2.0]])
    resumed = FeatureStore(prefix, manifest, rows, "clip", 2, 8, "static")
    assert resumed.pending_indices() == [1]
    assert np.allclose(resumed.features[0], [1.0, 2.0])


def test_qwen_feature_shards_are_disjoint_and_complete():
    indices = list(range(11))
    shards = [select_shard(indices, 3, shard) for shard in range(3)]
    assert sorted(index for shard in shards for index in shard) == indices
    assert not set(shards[0]) & set(shards[1])


def test_videochat3_metadata_preserves_exact_eight_frames():
    video = np.zeros((8, 12, 16, 3), dtype=np.uint8)
    assert video_metadata(video) == {
        "total_num_frames": 8,
        "fps": 1.0,
        "duration": 8.0,
        "width": 16,
        "height": 12,
    }


def test_grouped_probe_selection_never_splits_one_video_across_folds():
    labels = np.asarray([index % 2 for index in range(12)])
    groups = np.asarray([
        "video_{}".format(index // 2) for index in range(12)
    ])
    for fit_indices, validation_indices in grouped_cv_splits(
        labels, groups, folds=3, seed=7
    ):
        assert set(groups[fit_indices]).isdisjoint(groups[validation_indices])


def test_one_standard_error_rule_prefers_stronger_regularization():
    grid = {
        "1e-06": [
            {"macro_f1": 0.71, "accuracy": 0.72},
            {"macro_f1": 0.68, "accuracy": 0.70},
            {"macro_f1": 0.71, "accuracy": 0.73},
        ],
        "1e-05": [
            {"macro_f1": 0.695, "accuracy": 0.71},
            {"macro_f1": 0.695, "accuracy": 0.71},
            {"macro_f1": 0.695, "accuracy": 0.71},
        ],
        "0.0001": [
            {"macro_f1": 0.60, "accuracy": 0.65},
            {"macro_f1": 0.60, "accuracy": 0.65},
            {"macro_f1": 0.60, "accuracy": 0.65},
        ],
    }
    selected, summary = select_alpha_one_standard_error(
        grid, [1e-6, 1e-5, 1e-4]
    )
    assert selected == 1e-5
    assert summary["best_alpha_by_mean_macro_f1"] == 1e-6
    assert summary["rule"] == "one_standard_error_then_strongest_regularization"


def test_rq1_aggregate_reports_retention_and_worst_shift():
    shifts = {}
    for index, shift in enumerate(SHIFT_CONFIGS):
        source = 0.8
        target = 0.4 + 0.01 * index
        shifts[shift] = {
            "source": {
                "accuracy": source,
                "macro_recall": source,
                "macro_f1": source,
                "success_rate": 1.0,
            },
            "target": {
                "accuracy": target,
                "macro_recall": target,
                "macro_f1": target,
                "success_rate": 1.0,
            },
        }
    aggregate = aggregate_shifts(shifts)
    assert np.isclose(aggregate["macro_f1"]["source_mean"], 0.8)
    assert np.isclose(aggregate["macro_f1"]["target_worst_shift"], 0.4)
    assert np.isclose(
        aggregate["macro_f1"]["retention"],
        aggregate["macro_f1"]["target_mean"] / 0.8,
    )
    assert aggregate["validity"]["all_predictions_present"]


def test_rq1_split_audit_rejects_video_overlap():
    rows = []
    for shift in SHIFT_CONFIGS:
        for role, video_id in (
            ("source_train", "shared"),
            ("source_val", "shared"),
            ("target_val", "target"),
        ):
            for class_id in range(10):
                rows.append({
                    "video_id": "{}_{}".format(video_id, class_id),
                    "gt_label_id": class_id,
                    "memberships": [{"shift": shift, "role": role}],
                })
    valid, findings = audit_splits(rows)
    assert not valid
    assert findings["S1"]["video_overlaps"]["source_train_vs_source_val"]

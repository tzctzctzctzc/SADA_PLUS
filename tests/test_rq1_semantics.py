import json
from types import SimpleNamespace

import numpy as np
from PIL import Image

from tools.build_rq1_semantic_manifest import midpoint_labels
from tools.evaluate_rq1_semantics import confusion_metrics, jensen_shannon
from tools.merge_and_shard_rq1 import merge_rows, write_jsonl_atomic
from tools import prepare_rq1_frames
from tools.run_rq1_slowfast_probe import segment_feature
from tools.rq1_feature_store import FeatureStore, normalize, summarize_frame_features
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

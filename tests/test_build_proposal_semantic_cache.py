import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools" / "build_proposal_semantic_cache.py"
)
SPEC = importlib.util.spec_from_file_location(
    "build_proposal_semantic_cache", MODULE_PATH
)
cache_builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cache_builder)
build_cache = cache_builder.build_cache


def write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_builder_retains_frozen_geometry_and_drops_manifest_gt(tmp_path):
    manifest = tmp_path / "source_train_manifest.jsonl"
    write_jsonl(manifest, [
        {
            "candidate_id": "v-p1",
            "video_id": "v",
            "absolute_feature_index": 11,
            "absolute_center_time": 1.5,
            "proposal_start_sec": 1.0,
            "proposal_end_sec": 2.0,
            "source_fpn_levels": [0],
            "gt_assigned_class_ids": [3],
            "semantic_gt_kind": "KNOWN_ACTION",
        },
        {
            "candidate_id": "v-p2",
            "video_id": "v",
            "absolute_feature_index": 12,
            "absolute_center_time": 2.5,
            "proposal_start_sec": 2.0,
            "proposal_end_sec": 3.0,
            "source_fpn_levels": [0],
        },
    ])
    (tmp_path / "source_train_manifest.jsonl.meta.json").write_text(json.dumps({
        "domain": "source",
        "selection_teacher_provenance": {"weights": "ema"}
    }), encoding="utf-8")
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(predictions, [
        {
            "candidate_id": "v-p1",
            "video_id": "v",
            "absolute_feature_index": 11,
            "status": "ok",
            "prediction": "OPEN",
            "model": "test-vlm",
            "prompt_version": "v1",
            "prompt_sha256": "a" * 64,
            "base_prompt_sha256": "b" * 64,
            "wire_api": "test-api",
            "class_probabilities": [0.01, 0.01, 0.01, 0.91] + [0.01] * 6,
        },
        {
            "candidate_id": "v-p2",
            "video_id": "v",
            "absolute_feature_index": 12,
            "status": "ok",
            "prediction": "PUT",
            "model": "test-vlm",
            "prompt_version": "v1",
            "prompt_sha256": "c" * 64,
            "base_prompt_sha256": "b" * 64,
            "wire_api": "test-api",
        },
    ])

    payload = build_cache(manifest, [predictions])

    assert payload["selection_uses_gt"] is False
    assert payload["domain"] == "source"
    assert payload["split"] == "train"
    assert payload["entries"][0]["semantic_label"] == "OPEN"
    assert payload["entries"][0]["class_id"] == 3
    assert payload["entries"][0]["class_probabilities"][3] == pytest.approx(0.91)
    assert payload["entries"][0]["confidence"] == pytest.approx(0.91)
    assert "gt_assigned_class_ids" not in payload["entries"][0]
    assert payload["prompt_protocol"] == {
        "model": "test-vlm",
        "prompt_version": "v1",
        "prompt_sha256": "b" * 64,
        "wire_api": "test-api",
    }


def test_builder_rejects_prediction_identity_mismatch(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(manifest, [{
        "candidate_id": "v-p1",
        "video_id": "v",
        "absolute_feature_index": 11,
        "absolute_center_time": 1.5,
        "proposal_start_sec": 1.0,
        "proposal_end_sec": 2.0,
        "source_fpn_levels": [0],
    }])
    (tmp_path / "manifest.jsonl.meta.json").write_text(json.dumps({
        "domain": "target",
        "split": "train",
        "selection_teacher_provenance": {}
    }), encoding="utf-8")
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(predictions, [{
        "candidate_id": "v-p1",
        "video_id": "other",
        "absolute_feature_index": 11,
        "status": "ok",
        "prediction": "OPEN",
        "model": "test-vlm",
        "prompt_version": "v1",
        "prompt_sha256": "a" * 64,
    }])

    with pytest.raises(ValueError, match="identity"):
        build_cache(manifest, [predictions])


def test_builder_filters_by_observable_teacher_score(tmp_path):
    manifest = tmp_path / "target_train_manifest.jsonl"
    write_jsonl(manifest, [
        {
            "candidate_id": "v-p1",
            "video_id": "v",
            "absolute_feature_index": 11,
            "absolute_center_time": 1.5,
            "proposal_start_sec": 1.0,
            "proposal_end_sec": 2.0,
            "source_fpn_levels": [0],
            "teacher_score": 0.44,
        },
        {
            "candidate_id": "v-p2",
            "video_id": "v",
            "absolute_feature_index": 12,
            "absolute_center_time": 2.5,
            "proposal_start_sec": 2.0,
            "proposal_end_sec": 3.0,
            "source_fpn_levels": [1],
            "teacher_score": 0.45,
        },
    ])
    (tmp_path / "target_train_manifest.jsonl.meta.json").write_text(json.dumps({
        "domain": "target",
        "split": "train",
        "selection_teacher_provenance": {"weights": "ema"},
    }), encoding="utf-8")
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(predictions, [
        {
            "candidate_id": candidate_id,
            "video_id": "v",
            "absolute_feature_index": index,
            "status": "ok",
            "prediction": "OPEN",
            "model": "test-vlm",
            "prompt_version": "v1",
            "prompt_sha256": "a" * 64,
            "wire_api": "test-api",
        }
        for candidate_id, index in (("v-p1", 11), ("v-p2", 12))
    ])

    payload = build_cache(
        manifest,
        [predictions],
        min_teacher_score=0.45,
    )

    assert [entry["candidate_id"] for entry in payload["entries"]] == ["v-p2"]
    assert payload["entries"][0]["teacher_score"] == pytest.approx(0.45)
    assert payload["selection_filter"] == {
        "min_teacher_score": 0.45,
        "uses_target_gt": False,
    }

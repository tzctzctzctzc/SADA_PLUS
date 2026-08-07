import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tools.build_vlm_training_cache import build_entries, build_payload
from tools.export_vlm_training_candidates import limit_queries
from tools.migrate_legacy_candidate_blind_semantics import migrate_payload


def selection_provenance():
    return {
        "selection_teacher_checkpoint_sha256": "c" * 64,
        "selection_teacher_state_sha256": "d" * 64,
        "selection_teacher_weights": "ema",
        "selection_beta": 0.1,
        "selection_alpha": 0.7,
    }


def manifest_row(candidate_id="p123", position=123, center=12.5):
    return {
        "candidate_id": candidate_id,
        "video_id": "video",
        "absolute_feature_index": position,
        "absolute_center_time": center,
        "center_sec": center,
    }


def prediction_row(manifest, prediction="TARGET_ACTION"):
    return {
        "candidate_id": manifest["candidate_id"],
        "semantic_cache_schema_version": "offline-vlm-semantic-evidence-v1",
        "prompt_sha256": "a" * 64,
        "model": "qwen-test",
        "wire_api": "chat",
        "status": "ok",
        "prediction": prediction,
        "semantic_pass": prediction == "TARGET_ACTION",
        "model_reported_label": prediction,
        "parse_error": False,
        "protocol_error": False,
        "consistency_error": False,
        "video_id": manifest["video_id"],
        "absolute_feature_index": manifest["absolute_feature_index"],
        "absolute_center_time": manifest["absolute_center_time"],
        "center_sec": manifest["center_sec"],
    }


def test_query_budget_keeps_separated_highest_scores_per_video():
    rows = [
        {
            "candidate_id": "q{}".format(index),
            "video_id": "video",
            "center_sec": center,
            "sada_score_max": score,
        }
        for index, (center, score) in enumerate(
            [(1.0, 0.49), (1.4, 0.48), (3.0, 0.40), (5.0, 0.30)]
        )
    ]
    selected = limit_queries(rows, max_per_video=2, min_gap_sec=1.0)
    assert {row["center_sec"] for row in selected} == {1.0, 3.0}


@pytest.mark.parametrize(
    "prediction, expected",
    [
        ("TARGET_ACTION", True),
        ("OTHER_ACTIVITY", False),
        ("TRUE_BACKGROUND", False),
        ("UNCERTAIN", False),
    ],
)
def test_cache_contains_semantics_only(prediction, expected):
    manifest = manifest_row()
    entry = build_entries([manifest], [prediction_row(manifest, prediction)])[0]
    assert entry == {
        "video_id": "video",
        "absolute_feature_index": 123,
        "absolute_center_time": 12.5,
        "semantic_label": prediction,
        "semantic_pass": expected,
    }
    assert not any("teacher" in key or "structure" in key for key in entry)


def test_payload_locks_absolute_index_schema_and_provenance():
    manifest = manifest_row()
    payload = build_payload(
        [manifest], [prediction_row(manifest)], selection_provenance()
    )
    assert payload["version"] == 6
    assert payload["semantic_cache_schema_version"] == "offline-vlm-semantic-evidence-v1"
    assert payload["source"] == "offline_vlm_candidate_blind_semantics"
    assert payload["key_fields"] == ["video_id", "absolute_feature_index"]
    assert payload["counts"] == {"TARGET_ACTION": 1}


def test_old_dual_gate_prediction_schema_is_rejected():
    manifest = manifest_row()
    result = prediction_row(manifest)
    result.pop("semantic_cache_schema_version")
    result["routing_schema_version"] = "semantic-structural-exact-anchor-routing-v3"
    with pytest.raises(ValueError, match="unsupported semantic prediction schema"):
        build_entries([manifest], [result])


def test_silent_semantic_inconsistency_is_rejected():
    manifest = manifest_row()
    result = prediction_row(manifest)
    result["semantic_pass"] = False
    with pytest.raises(ValueError, match="semantic_pass"):
        build_entries([manifest], [result])


def test_absolute_index_is_exact_and_unique():
    first = manifest_row("first", position=10)
    duplicate = manifest_row("second", position=10, center=13.0)
    with pytest.raises(ValueError, match="duplicate semantic cache key"):
        build_entries(
            [first, duplicate],
            [prediction_row(first), prediction_row(duplicate)],
        )

    bad = copy.deepcopy(first)
    bad["absolute_feature_index"] = 10.25
    with pytest.raises(ValueError, match="absolute_feature_index"):
        build_entries([bad], [prediction_row(bad)])


def test_prediction_coverage_must_equal_manifest():
    first = manifest_row("first", 10)
    second = manifest_row("second", 11)
    with pytest.raises(ValueError, match="coverage differs"):
        build_entries([first, second], [prediction_row(first)])


def legacy_rows():
    manifest = {
        "candidate_id": "video-l02-g0000042",
        "video_id": "video",
        "fpn_level": 2,
        "anchor_index": 10,
        "global_anchor_index": 42,
        "point_stride": 4.0,
        "absolute_center_time": 22.0,
        "center_sec": 22.0,
        "teacher_top1_class": 5,
        "teacher_structural_eligible": True,
    }
    prediction = {
        "candidate_id": manifest["candidate_id"],
        "video_id": manifest["video_id"],
        "fpn_level": manifest["fpn_level"],
        "anchor_index": manifest["anchor_index"],
        "global_anchor_index": manifest["global_anchor_index"],
        "absolute_center_time": manifest["absolute_center_time"],
        "center_sec": manifest["center_sec"],
        "routing_schema_version": "candidate-blind-exact-anchor-routing-v1",
        "prompt_version": "candidate-blind-target-foreground-gate-v1",
        "prompt_sha256": "b" * 64,
        "model": "qwen-test",
        "wire_api": "chat",
        "status": "ok",
        "prediction": "TARGET_FOREGROUND",
        "parse_error": False,
        "protocol_error": False,
        "consistency_error": False,
        "teacher_top1_class": 5,
        "route": "action",
    }
    return manifest, prediction


def test_legacy_candidate_blind_migration_keeps_only_absolute_semantics():
    manifest, prediction = legacy_rows()
    payload = migrate_payload(
        [manifest],
        [prediction],
        "c" * 64,
        "d" * 64,
        selection_provenance(),
    )
    assert payload["counts"] == {"TARGET_ACTION": 1}
    assert payload["migration"]["input_predictions_sha256"] == "d" * 64
    assert payload["entries"] == [
        {
            "video_id": "video",
            "absolute_feature_index": 40,
            "absolute_center_time": 22.0,
            "semantic_label": "TARGET_ACTION",
            "semantic_pass": True,
        }
    ]


def test_legacy_candidate_blind_migration_rejects_coordinate_mismatch():
    manifest, prediction = legacy_rows()
    prediction["anchor_index"] += 1
    with pytest.raises(ValueError, match="anchor_index differs"):
        migrate_payload(
            [manifest],
            [prediction],
            "c" * 64,
            "d" * 64,
            selection_provenance(),
        )


def test_legacy_candidate_blind_migration_cli(tmp_path):
    manifest, prediction = legacy_rows()
    manifest_path = tmp_path / "manifest.jsonl"
    prediction_path = tmp_path / "predictions.jsonl"
    output_path = tmp_path / "cache.json"
    checkpoint_path = tmp_path / "teacher.pt"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    prediction_path.write_text(json.dumps(prediction) + "\n", encoding="utf-8")
    torch.save(
        {"ema_state_dict": {"weight": torch.tensor([1.0])}}, checkpoint_path
    )
    Path(str(manifest_path) + ".meta.json").write_text(
        json.dumps(
            {
                "selection_uses_target_gt": False,
                "checkpoint": str(checkpoint_path),
                "weights": "ema",
                "beta": 0.1,
                "alpha": 0.7,
            }
        ),
        encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable,
            str(
                Path(__file__).parents[1]
                / "tools"
                / "migrate_legacy_candidate_blind_semantics.py"
            ),
            "--manifest",
            str(manifest_path),
            "--predictions",
            str(prediction_path),
            "--output",
            str(output_path),
        ],
        check=True,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert json.loads(output_path.read_text(encoding="utf-8"))["counts"] == {
        "TARGET_ACTION": 1
    }

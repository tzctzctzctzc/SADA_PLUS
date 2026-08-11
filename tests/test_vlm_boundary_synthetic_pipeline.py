import json
import subprocess
import sys
from pathlib import Path


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def run(root, *args):
    result = subprocess.run(
        [sys.executable, *map(str, args)],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_tiny_synthetic_coarse_fine_score_pipeline(tmp_path):
    root = Path(__file__).resolve().parents[1]
    proposal = {
        "proposal_id": "v1:l0:x4:a4:c1",
        "video_id": "v1",
        "video_duration_seconds": 10.0,
        "predicted_class_id": 1,
        "predicted_class_name": "PUT",
        "score": 0.9,
        "coarse_segment_seconds": [2.0, 8.0],
        "origin_stride_seconds": 0.6,
        "fps": 30.0,
        "posthoc_gt": {
            "kind": "VALID_ACTION",
            "matched_segment_seconds": [3.0, 7.0],
        },
    }
    proposals = tmp_path / "proposals.jsonl"
    coarse_manifest = tmp_path / "coarse.jsonl"
    coarse_predictions = tmp_path / "coarse_predictions.jsonl"
    fine_manifest = tmp_path / "fine.jsonl"
    fine_predictions = tmp_path / "fine_predictions.jsonl"
    annotations = tmp_path / "annotations.json"
    report_path = tmp_path / "report.json"
    write_jsonl(proposals, [proposal])
    annotations.write_text(json.dumps({
        "database": {
            "v1": {
                "subset": "synthetic",
                "annotations": [{"segment": [3.0, 7.0], "label_id": 1}],
            }
        }
    }), encoding="utf-8")

    run(
        root,
        root / "tools" / "build_vlm_boundary_manifest.py",
        "coarse", "--proposals", proposals, "--output", coarse_manifest,
        "--task-mode", "boundary-only",
    )
    coarse = json.loads(coarse_manifest.read_text(encoding="utf-8").splitlines()[0])
    start_option = min(
        coarse["start_candidates"], key=lambda item: abs(item["timestamp_sec"] - 3.0)
    )["option"]
    end_option = min(
        coarse["end_candidates"], key=lambda item: abs(item["timestamp_sec"] - 7.0)
    )["option"]
    coarse_decision = {
        "instance_description": "right hand puts the same bowl on the counter",
        "start_decision": "SELECT_CANDIDATE",
        "selected_start_option": start_option,
        "end_decision": "SELECT_CANDIDATE",
        "selected_end_option": end_option,
        "reason": "synthetic",
    }
    write_jsonl(coarse_predictions, [{
        "query_id": coarse["query_id"],
        "run_status": "ok",
        "decision": coarse_decision,
    }])
    run(
        root,
        root / "tools" / "build_vlm_boundary_manifest.py",
        "fine", "--proposals", proposals, "--output", fine_manifest,
        "--coarse-manifest", coarse_manifest,
        "--coarse-predictions", coarse_predictions,
    )
    fine_rows = [json.loads(line) for line in fine_manifest.read_text(encoding="utf-8").splitlines()]
    assert {row["boundary_side"] for row in fine_rows} == {"start", "end"}
    write_jsonl(fine_predictions, [
        {
            "query_id": row["query_id"],
            "run_status": "ok",
            "decision": {
                "decision": "KEEP_COARSE",
                "selected_option": None,
                "reason": "synthetic keep",
            },
        }
        for row in fine_rows
    ])
    run(
        root,
        root / "tools" / "score_vlm_boundary_audit.py",
        "--proposals", proposals,
        "--coarse-manifest", coarse_manifest,
        "--coarse-predictions", coarse_predictions,
        "--fine-manifest", fine_manifest,
        "--fine-predictions", fine_predictions,
        "--annotation-json", annotations,
        "--official-split", "synthetic",
        "--output", report_path,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["protocol"].endswith("v5")
    assert report["fixed_proposal_invariants"]["coarse_keep_sada_policy"] == (
        "terminal_no_fine_reopening"
    )
    assert report["task_mode"] == "boundary-only"
    assert report["validity"]["applicable"] is False
    assert all(
        variant["evaluator_parity"]["passed"]
        for variant in report["variants"].values()
    )
    sides = report["side_outputs"]["vlm_refined"][proposal["proposal_id"]]
    assert sides["start"]["source"] == "COARSE_KEEP"
    assert sides["end"]["source"] == "COARSE_KEEP"
    assert report["oracle_candidate_spaces"][proposal["proposal_id"]]["choices"] == [
        "SELECT_CANDIDATE", "KEEP_SADA", "KEEP_COARSE",
        "SIDE_ABSTENTION_TO_DETECTION_FALLBACK"
    ]

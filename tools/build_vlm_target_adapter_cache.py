import argparse
import hashlib
import json
import math
from pathlib import Path

from tools.calibrate_vlm_temperature import (
    SCHEMA_VERSION as TEMPERATURE_CALIBRATION_SCHEMA_VERSION,
    row_class_logits,
    temperature_scaled_probabilities,
)


KNOWN = (
    "TAKE", "PUT", "WASH", "OPEN", "CLOSE", "INSERT", "TURN_ON",
    "CUT", "TURN_OFF", "POUR",
)


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions", required=True, nargs="+")
    parser.add_argument("--source-gate-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--temperature-calibration-report")
    parser.add_argument(
        "--allow-gate-failure",
        action="store_true",
        help="explicitly build a diagnostic cache when a non-CLIP source gate fails",
    )
    args = parser.parse_args()

    gate = json.loads(Path(args.source_gate_report).read_text(encoding="utf-8"))
    if gate.get("gate_version") != "vlm-target-adapter-source-gate-v1":
        raise ValueError("invalid source gate report")
    manifest_rows = read_jsonl(args.manifest)
    manifest = {row["candidate_id"]: row for row in manifest_rows}
    if len(manifest) != len(manifest_rows):
        raise ValueError("duplicate target manifest candidate")
    prediction_rows = [
        row for path in args.predictions for row in read_jsonl(path)
        if row.get("status") == "ok"
    ]
    predictions = {row["candidate_id"]: row for row in prediction_rows}
    if len(predictions) != len(prediction_rows):
        raise ValueError("duplicate target VLM prediction")
    missing = set(manifest) - set(predictions)
    if missing:
        raise ValueError("target VLM predictions are incomplete")
    protocols = {
        (
            row["prompt_version"],
            row.get("base_prompt_sha256", row["prompt_sha256"]),
            row["model"], row["wire_api"],
        )
        for row in predictions.values()
    }
    if len(protocols) != 1:
        raise ValueError("target VLM predictions use mixed protocols")
    prompt_version, prompt_sha256, model, wire_api = protocols.pop()
    is_clip = wire_api == "openai-clip"
    if not gate.get("adapter_pass") and not is_clip and not args.allow_gate_failure:
        raise ValueError("source VLM target adapter gate did not pass")
    meta = json.loads(
        Path(args.manifest + ".meta.json").read_text(encoding="utf-8")
    )
    if meta.get("domain") != "target" or meta.get("selection_uses_gt") is not False:
        raise ValueError("training cache requires a GT-free target manifest")

    calibration = None
    calibration_path = args.temperature_calibration_report
    if calibration_path:
        calibration = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
        if calibration.get("schema_version") != TEMPERATURE_CALIBRATION_SCHEMA_VERSION:
            raise ValueError("invalid temperature calibration report")
        if calibration.get("selection_uses_target_gt") is not False:
            raise ValueError("temperature calibration must be source-only")
        expected_protocol = {
            "model": model,
            "prompt_version": prompt_version,
            "prompt_sha256": prompt_sha256,
            "wire_api": wire_api,
        }
        if calibration.get("source_protocol") != expected_protocol:
            raise ValueError("temperature calibration protocol differs from target predictions")
        temperature = float(calibration["temperature"])
        if temperature <= 0:
            raise ValueError("temperature calibration must be positive")

    entries = []
    for candidate_id, row in manifest.items():
        label = predictions[candidate_id]["prediction"]
        if label not in KNOWN:
            continue
        entry = {
            "video_id": row["video_id"],
            "absolute_feature_index": row["absolute_feature_index"],
            "absolute_center_time": row["absolute_center_time"],
            "proposal_start_sec": row["proposal_start_sec"],
            "proposal_end_sec": row["proposal_end_sec"],
            "source_fpn_levels": row["source_fpn_levels"],
            "semantic_label": label,
            "class_id": KNOWN.index(label),
        }
        prediction = predictions[candidate_id]
        probabilities = prediction.get("class_probabilities")
        if calibration is not None:
            probabilities = temperature_scaled_probabilities(
                row_class_logits(prediction), temperature
            )
        if probabilities is not None:
            if not isinstance(probabilities, list) or len(probabilities) != len(KNOWN):
                raise ValueError("VLM prediction has invalid class probabilities")
            if any(
                not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0.0
                for value in probabilities
            ):
                raise ValueError("VLM prediction has invalid class probabilities")
            if abs(sum(probabilities) - 1.0) > 1e-4:
                raise ValueError("VLM prediction probabilities must sum to one")
            if max(range(len(probabilities)), key=probabilities.__getitem__) != entry["class_id"]:
                raise ValueError("VLM prediction probabilities and class id differ")
            entry["class_probabilities"] = probabilities
            order = sorted(probabilities, reverse=True)
            entry["confidence"] = order[0]
            entry["margin"] = order[0] - order[1]
        entries.append(entry)
    payload = {
        "schema_version": (
            "offline-clip-target-adapter-v3"
            if is_clip else "offline-vlm-target-adapter-v2"
        ),
        "source": "offline_vlm_proposal_semantics",
        "key_fields": ["video_id", "absolute_feature_index"],
        "selection_teacher_provenance": meta["selection_teacher_provenance"],
        "prompt_protocol": {
            "prompt_version": prompt_version,
            "prompt_sha256": prompt_sha256,
            "model": model,
            "wire_api": wire_api,
        },
        "training_policy": (
            "runtime_selectable_clip_class_policy"
            if is_clip else "known_class_and_current_teacher_top1_agreement"
        ),
        "background_enabled": False,
        "entries": entries,
        "provenance": {
            "manifest_sha256": sha256(args.manifest),
            "prediction_sha256": [sha256(path) for path in args.predictions],
            "source_gate_report_sha256": sha256(args.source_gate_report),
            "source_gate_pass": bool(gate.get("adapter_pass")),
            "source_gate_override": bool(args.allow_gate_failure and not gate.get("adapter_pass")),
            "temperature_calibration_report_sha256": (
                sha256(calibration_path) if calibration_path else None
            ),
        },
        "counts": {
            "manifest": len(manifest),
            "known_entries": len(entries),
            "abstained_entries": len(manifest) - len(entries),
        },
    }
    if calibration is not None:
        payload["probability_calibration"] = {
            "schema_version": calibration["schema_version"],
            "source_predictions_sha256": calibration["source_predictions_sha256"],
            "known_action_rows": calibration["known_action_rows"],
            "temperature": calibration["temperature"],
            "before": calibration["before"],
            "after": calibration["after"],
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

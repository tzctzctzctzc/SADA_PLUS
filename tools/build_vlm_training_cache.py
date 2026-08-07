"""Validate dual-gate exact-anchor routes and serialize a training cache."""

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path


ROUTING_SCHEMA_VERSION = "semantic-structural-exact-anchor-routing-v3"
ROUTING_POLICY = "exact_anchor_rescue"
ROUTING_MODES = (
    "semantic_only",
    "structure_only",
    "semantic_and_structure",
)
ACTION_PREDICTION = "TARGET_ACTION"
NON_ACTION_PREDICTIONS = {"OTHER_ACTIVITY", "TRUE_BACKGROUND", "UNCERTAIN"}
EXACT_KEY_FIELDS = (
    "video_id",
    "fpn_level",
    "global_anchor_index",
    "absolute_center_time",
    "teacher_top1_class",
)
STRUCTURE_FIELDS = (
    "structure_probe_schema_version",
    "structure_probe_sha256",
    "structure_score",
    "structure_threshold",
    "structural_pass",
    "structure_feature_values",
)
PROTOCOL_FIELDS = {
    "routing_schema_version",
    "prompt_sha256",
    "model",
    "wire_api",
    "status",
    "prediction",
    "semantic_pass",
    "structural_pass",
    "ablation_routes",
    "route",
    "action_rescue",
    "pseudo_label",
    "model_reported_label",
    "parse_error",
    "protocol_error",
    "consistency_error",
} | set(STRUCTURE_FIELDS)


def read_jsonl(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError("JSONL file not found: {}".format(path))
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("JSONL file is empty: {}".format(path))
    return rows


def _absolute_center_time(row):
    absolute = float(row["absolute_center_time"])
    alias = float(row["center_sec"])
    if not math.isfinite(absolute) or absolute < 0:
        raise ValueError("invalid absolute_center_time")
    if not math.isfinite(alias) or abs(alias - absolute) > 1e-6:
        raise ValueError("center_sec differs from absolute_center_time")
    return absolute


def _strict_key(row):
    return (
        str(row["video_id"]),
        int(row["fpn_level"]),
        int(row["global_anchor_index"]),
        float(row["absolute_center_time"]),
        int(row["teacher_top1_class"]),
    )


def _source_structure(row):
    if not set(STRUCTURE_FIELDS).issubset(row):
        raise ValueError("exact-anchor row is missing source-supervised structure")
    if row["structure_probe_schema_version"] != "source-supervised-anchor-validity-v1":
        raise ValueError("unsupported structure probe schema")
    if not re.fullmatch(r"[0-9a-f]{64}", row["structure_probe_sha256"]):
        raise ValueError("invalid structure probe sha256")
    score = float(row["structure_score"])
    threshold = float(row["structure_threshold"])
    features = row["structure_feature_values"]
    if (
        not math.isfinite(score)
        or not math.isfinite(threshold)
        or not 0.0 <= score <= 1.0
        or not 0.0 <= threshold <= 1.0
        or not isinstance(features, list)
        or len(features) != 13
        or not all(math.isfinite(float(value)) for value in features)
    ):
        raise ValueError("invalid source-supervised structure values")
    passed = score >= threshold
    if row["structural_pass"] is not passed:
        raise ValueError("structural_pass disagrees with source-supervised score")
    return passed


def _validate_manifest(manifest_rows):
    by_id = {}
    keys = set()
    structure_signatures = set()
    for row in manifest_rows:
        required = set(EXACT_KEY_FIELDS) | {
            "candidate_id", "anchor_index", "center_sec", "member_anchors"
        } | set(STRUCTURE_FIELDS)
        if not required.issubset(row):
            raise ValueError("exact-anchor manifest is missing a strict field")
        candidate_id = row["candidate_id"]
        if candidate_id in by_id:
            raise ValueError("duplicate candidate_id: {}".format(candidate_id))
        _absolute_center_time(row)
        top1 = int(row["teacher_top1_class"])
        if top1 < 0 or top1 >= 10:
            raise ValueError("invalid teacher_top1_class")
        members = row["member_anchors"]
        if not isinstance(members, list) or len(members) != 1:
            raise ValueError("exact-anchor query must have exactly one member")
        member = members[0]
        _source_structure(row)
        _source_structure(member)
        if (
            int(member["fpn_level"]) != int(row["fpn_level"])
            or int(member["anchor_index"]) != int(row["anchor_index"])
            or int(member["global_anchor_index"]) != int(row["global_anchor_index"])
            or abs(float(member["absolute_center_time"]) - float(row["absolute_center_time"])) > 1e-6
            or int(member["teacher_top1_class"]) != top1
            or any(member[field] != row[field] for field in STRUCTURE_FIELDS)
        ):
            raise ValueError("exact-anchor member does not match its query")
        key = _strict_key(row)
        if key in keys:
            raise ValueError("duplicate exact-anchor key: {}".format(key))
        keys.add(key)
        structure_signatures.add(
            (row["structure_probe_schema_version"], row["structure_probe_sha256"], float(row["structure_threshold"]))
        )
        by_id[candidate_id] = row
    if not by_id:
        raise ValueError("manifest is empty")
    if len(structure_signatures) != 1:
        raise ValueError("manifest mixes structure probes or thresholds")
    return by_id, structure_signatures.pop()


def _validate_provenance(prediction_rows):
    signatures = set()
    for row in prediction_rows:
        if row.get("routing_schema_version") != ROUTING_SCHEMA_VERSION:
            raise ValueError("unsupported routing_schema_version")
        prompt_sha256 = row.get("prompt_sha256")
        if not isinstance(prompt_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", prompt_sha256):
            raise ValueError("invalid prompt_sha256")
        model = row.get("model")
        wire_api = row.get("wire_api")
        if not isinstance(model, str) or not model:
            raise ValueError("missing model")
        if wire_api not in {"chat", "responses"}:
            raise ValueError("invalid wire_api")
        signatures.add((prompt_sha256, model, wire_api))
    if len(signatures) != 1:
        raise ValueError("prediction file mixes Prompt, model, or wire API")
    prompt_sha256, model, wire_api = signatures.pop()
    return {
        "routing_schema_version": ROUTING_SCHEMA_VERSION,
        "prompt_sha256": prompt_sha256,
        "model": model,
        "wire_api": wire_api,
    }


def _expected_routes(semantic_pass, structural_pass):
    return {
        "semantic_only": "action" if semantic_pass else "ignore",
        "structure_only": "action" if structural_pass else "ignore",
        "semantic_and_structure": (
            "action" if semantic_pass and structural_pass else "ignore"
        ),
    }


def _validate_prediction(prediction, manifest):
    if not PROTOCOL_FIELDS.issubset(prediction):
        raise ValueError("prediction is missing a routing protocol field")
    if prediction["status"] != "ok":
        raise ValueError("latest prediction row is not successful")
    if prediction.get("candidate_id") != manifest["candidate_id"]:
        raise ValueError("prediction candidate_id differs from manifest")
    _absolute_center_time(prediction)
    if _strict_key(prediction) != _strict_key(manifest):
        raise ValueError("prediction exact-anchor key differs from manifest")
    if prediction.get("anchor_index") != manifest["anchor_index"]:
        raise ValueError("prediction anchor_index differs from manifest")
    if any(prediction[field] is not False for field in ("parse_error", "protocol_error", "consistency_error")):
        raise ValueError("prediction contains a protocol or parsing error")
    for field in STRUCTURE_FIELDS:
        if prediction[field] != manifest[field]:
            raise ValueError("prediction structure differs from manifest")
    structural_pass = _source_structure(manifest)
    if prediction["structural_pass"] is not structural_pass:
        raise ValueError("prediction structural_pass differs from manifest")

    label = prediction["prediction"]
    if label not in {ACTION_PREDICTION} | NON_ACTION_PREDICTIONS:
        raise ValueError("legacy or unsupported prediction label")
    semantic_pass = label == ACTION_PREDICTION
    if prediction["semantic_pass"] is not semantic_pass:
        raise ValueError("semantic_pass disagrees with VLM prediction")
    local_uncertain = prediction.get("local_gate_reason") is not None
    if local_uncertain:
        if label != "UNCERTAIN" or prediction["model_reported_label"] is not None:
            raise ValueError("invalid local UNCERTAIN result")
    elif prediction["model_reported_label"] != label:
        raise ValueError("model label differs from normalized prediction")

    routes = _expected_routes(semantic_pass, structural_pass)
    if prediction["ablation_routes"] != routes:
        raise ValueError("runner ablation routes disagree with gate decisions")
    joint_action = routes["semantic_and_structure"] == "action"
    if (
        prediction["route"] != routes["semantic_and_structure"]
        or prediction["action_rescue"] is not joint_action
        or prediction["pseudo_label"] != (
            manifest["teacher_top1_class"] if joint_action else None
        )
    ):
        raise ValueError("runner joint route disagrees with dual-gate protocol")
    return routes


def build_entries(manifest_rows, prediction_rows, routing_mode="semantic_and_structure"):
    if routing_mode not in ROUTING_MODES:
        raise ValueError("unsupported routing mode: {}".format(routing_mode))
    manifest_by_id, _ = _validate_manifest(manifest_rows)
    _validate_provenance(prediction_rows)
    latest_predictions = {}
    for prediction in prediction_rows:
        candidate_id = prediction.get("candidate_id")
        if candidate_id not in manifest_by_id:
            raise ValueError("prediction candidate is absent from manifest")
        latest_predictions[candidate_id] = prediction
    if set(latest_predictions) != set(manifest_by_id):
        raise ValueError("prediction coverage differs from manifest")

    entries = []
    for candidate_id, manifest in manifest_by_id.items():
        routes = _validate_prediction(latest_predictions[candidate_id], manifest)
        decision = routes[routing_mode]
        entry = {
            "video_id": manifest["video_id"],
            "fpn_level": int(manifest["fpn_level"]),
            "anchor_index": int(manifest["anchor_index"]),
            "global_anchor_index": int(manifest["global_anchor_index"]),
            "absolute_center_time": float(manifest["absolute_center_time"]),
            "teacher_top1_class": int(manifest["teacher_top1_class"]),
            "decision": decision,
        }
        if decision == "action":
            entry["class_id"] = int(manifest["teacher_top1_class"])
        entries.append(entry)
    return entries


def build_payload(manifest_rows, prediction_rows, routing_mode="semantic_and_structure"):
    provenance = _validate_provenance(prediction_rows)
    _, structure_signature = _validate_manifest(manifest_rows)
    entries = build_entries(manifest_rows, prediction_rows, routing_mode)
    counts = Counter(entry["decision"] for entry in entries)
    return {
        "version": 5,
        "source": "offline_vlm_semantics_plus_source_supervised_anchor_validity",
        "routing_policy": ROUTING_POLICY,
        "routing_mode": routing_mode,
        **provenance,
        "structure_probe_schema_version": structure_signature[0],
        "structure_probe_sha256": structure_signature[1],
        "structure_threshold": structure_signature[2],
        "routing_rule": {
            "semantic_only": "VLM TARGET_ACTION",
            "structure_only": "source-supervised structural_pass",
            "semantic_and_structure": "both gates pass",
            "class": "always SADA teacher top-1",
            "valid_non_action": "ignore",
            "background": "forbidden",
        },
        "bank_key_fields": list(EXACT_KEY_FIELDS),
        "counts": {
            "action": counts.get("action", 0),
            "ignore": counts.get("ignore", 0),
            "background": 0,
        },
        "entries": entries,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--routing-mode", choices=ROUTING_MODES, default="semantic_and_structure")
    args = parser.parse_args()
    payload = build_payload(
        read_jsonl(args.manifest),
        read_jsonl(args.predictions),
        args.routing_mode,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "routing_mode": args.routing_mode, **payload["counts"]}, indent=2))


if __name__ == "__main__":
    main()

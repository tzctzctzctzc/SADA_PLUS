"""Validate candidate-blind exact-anchor VLM routes and serialize a cache."""

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path


ROUTING_SCHEMA_VERSION = "candidate-blind-exact-anchor-routing-v1"
ROUTING_POLICY = "exact_anchor_rescue"
ACTION_PREDICTION = "TARGET_FOREGROUND"
NON_ACTION_PREDICTIONS = {"OTHER_ACTIVITY", "TRUE_BACKGROUND", "UNCERTAIN"}
EXACT_KEY_FIELDS = (
    "video_id",
    "fpn_level",
    "global_anchor_index",
    "absolute_center_time",
    "teacher_top1_class",
)
PROTOCOL_FIELDS = {
    "routing_schema_version",
    "prompt_sha256",
    "model",
    "wire_api",
    "status",
    "prediction",
    "route",
    "action_rescue",
    "pseudo_label",
    "center_interaction",
    "target_signature",
    "model_reported_label",
    "parse_error",
    "protocol_error",
    "consistency_error",
}


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


def _validate_manifest(manifest_rows):
    by_id = {}
    keys = set()
    for row in manifest_rows:
        required = set(EXACT_KEY_FIELDS) | {
            "candidate_id",
            "anchor_index",
            "center_sec",
            "member_anchors",
        }
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
        if (
            int(member["fpn_level"]) != int(row["fpn_level"])
            or int(member["anchor_index"]) != int(row["anchor_index"])
            or int(member["global_anchor_index"])
            != int(row["global_anchor_index"])
            or abs(
                float(member["absolute_center_time"])
                - float(row["absolute_center_time"])
            )
            > 1e-6
            or int(member["teacher_top1_class"])
            != int(row["teacher_top1_class"])
        ):
            raise ValueError("exact-anchor member does not match its query")
        key = _strict_key(row)
        if key in keys:
            raise ValueError("duplicate exact-anchor key: {}".format(key))
        keys.add(key)
        by_id[candidate_id] = row
    if not by_id:
        raise ValueError("manifest is empty")
    return by_id


def _validate_provenance(prediction_rows):
    signatures = set()
    for row in prediction_rows:
        if row.get("routing_schema_version") != ROUTING_SCHEMA_VERSION:
            raise ValueError("unsupported routing_schema_version")
        prompt_sha256 = row.get("prompt_sha256")
        if not isinstance(prompt_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", prompt_sha256
        ):
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


def _expected_prediction(center_interaction, target_signature):
    if center_interaction == "YES" and target_signature == "YES":
        return ACTION_PREDICTION
    if center_interaction == "YES" and target_signature == "NO":
        return "OTHER_ACTIVITY"
    if center_interaction == "NO" and target_signature != "YES":
        return "TRUE_BACKGROUND"
    return "UNCERTAIN"


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
    if any(
        prediction[field] is not False
        for field in ("parse_error", "protocol_error", "consistency_error")
    ):
        raise ValueError("prediction contains a protocol or parsing error")

    center_interaction = prediction["center_interaction"]
    target_signature = prediction["target_signature"]
    if center_interaction not in {"YES", "NO", "UNCERTAIN"}:
        raise ValueError("invalid center_interaction")
    if target_signature not in {"YES", "NO", "UNCERTAIN"}:
        raise ValueError("invalid target_signature")
    expected = _expected_prediction(center_interaction, target_signature)
    if prediction["prediction"] != expected:
        raise ValueError("prediction disagrees with gate fields")

    is_action = (
        prediction["prediction"] == ACTION_PREDICTION
        and prediction["action_rescue"] is True
        and prediction["route"] == "action"
        and prediction["center_interaction"] == "YES"
        and prediction["target_signature"] == "YES"
        and prediction["pseudo_label"]
        == prediction["teacher_top1_class"]
        and prediction["pseudo_label"]
        == manifest["teacher_top1_class"]
        and prediction["model_reported_label"] == ACTION_PREDICTION
    )
    if prediction["action_rescue"] is True and not is_action:
        raise ValueError("action route does not satisfy the complete protocol")
    if is_action:
        return "action"

    if prediction["prediction"] not in NON_ACTION_PREDICTIONS:
        raise ValueError("legacy or unsupported prediction label")
    if (
        prediction["action_rescue"] is not False
        or prediction["route"] != "ignore"
        or prediction["pseudo_label"] is not None
    ):
        raise ValueError("non-confirmed prediction must route to ignore")

    local_uncertain = prediction.get("local_gate_reason") is not None
    if local_uncertain:
        if (
            prediction["prediction"] != "UNCERTAIN"
            or prediction["model_reported_label"] is not None
        ):
            raise ValueError("invalid local UNCERTAIN result")
    elif prediction["model_reported_label"] != prediction["prediction"]:
        raise ValueError("model label differs from normalized prediction")
    return "ignore"


def build_entries(manifest_rows, prediction_rows):
    manifest_by_id = _validate_manifest(manifest_rows)
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
    runner_action_keys = set()
    for candidate_id, manifest in manifest_by_id.items():
        prediction = latest_predictions[candidate_id]
        decision = _validate_prediction(prediction, manifest)
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
            runner_action_keys.add(_strict_key(prediction))
        entries.append(entry)

    cache_action_keys = {
        _strict_key(entry) for entry in entries if entry["decision"] == "action"
    }
    if runner_action_keys != cache_action_keys:
        raise ValueError("runner action keys differ from cache action keys")
    if any(entry["decision"] == "background" for entry in entries):
        raise ValueError("candidate-blind rescue must not produce background")
    return entries


def build_payload(manifest_rows, prediction_rows):
    provenance = _validate_provenance(prediction_rows)
    entries = build_entries(manifest_rows, prediction_rows)
    counts = Counter(entry["decision"] for entry in entries)
    return {
        "version": 3,
        "source": "offline_candidate_blind_exact_anchor_vlm_without_target_gt",
        "routing_policy": ROUTING_POLICY,
        **provenance,
        "routing_rule": {
            "validated_target_foreground": "action_with_teacher_top1",
            "valid_non_confirm_outputs": "ignore",
            "background": "forbidden",
        },
        "bank_key_fields": list(EXACT_KEY_FIELDS),
        "counts": {
            "action": counts.get("action", 0),
            "ignore": counts.get("ignore", 0),
            "background": counts.get("background", 0),
        },
        "entries": entries,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = build_payload(
        read_jsonl(args.manifest),
        read_jsonl(args.predictions),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), **payload["counts"]}, indent=2))


if __name__ == "__main__":
    main()

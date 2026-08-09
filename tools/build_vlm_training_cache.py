"""Validate candidate-blind VLM predictions and build an exact semantic cache."""

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path


LEGACY_PREDICTION_SCHEMA_VERSION = "offline-vlm-semantic-evidence-v1"
SEMANTIC_CACHE_SCHEMA_VERSION = "offline-vlm-bidirectional-semantic-evidence-v2"
LOW_PROMPT_ROLE = "low_rescue"
HIGH_PROMPT_ROLE = "high_veto"
PROMPT_ROLES = {LOW_PROMPT_ROLE, HIGH_PROMPT_ROLE}
ACTION_PREDICTION = "TARGET_ACTION"
VALID_PREDICTIONS = {
    ACTION_PREDICTION,
    "OTHER_ACTIVITY",
    "TRUE_BACKGROUND",
    "UNCERTAIN",
}
SELECTION_PROVENANCE_FIELDS = {
    "selection_teacher_checkpoint_sha256",
    "selection_teacher_state_sha256",
    "selection_teacher_weights",
    "selection_beta",
    "selection_alpha",
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


def _absolute_index(row):
    value = row["absolute_feature_index"]
    if isinstance(value, bool) or int(value) != value or int(value) < 0:
        raise ValueError("absolute_feature_index must be a non-negative integer")
    return int(value)


def _absolute_time(row):
    value = float(row["absolute_center_time"])
    alias = float(row["center_sec"])
    if not math.isfinite(value) or value < 0 or abs(value - alias) > 1e-6:
        raise ValueError("invalid or inconsistent absolute center time")
    return value


def _semantic_key(row):
    return _prompt_role(row), str(row["video_id"]), _absolute_index(row)


def _prompt_role(row):
    role = row.get("prompt_role", LOW_PROMPT_ROLE)
    if role not in PROMPT_ROLES:
        raise ValueError("invalid semantic prompt role")
    return role


def _validate_manifest(rows):
    by_id = {}
    keys = set()
    required = {
        "candidate_id",
        "video_id",
        "absolute_feature_index",
        "absolute_center_time",
        "center_sec",
    }
    for row in rows:
        if not required.issubset(row):
            raise ValueError("semantic manifest is missing an exact-coordinate field")
        candidate_id = str(row["candidate_id"])
        if candidate_id in by_id:
            raise ValueError("duplicate candidate_id: {}".format(candidate_id))
        _absolute_time(row)
        key = _semantic_key(row)
        if key in keys:
            raise ValueError("duplicate semantic cache key: {}".format(key))
        keys.add(key)
        by_id[candidate_id] = row
    if not by_id:
        raise ValueError("manifest is empty")
    return by_id


def _validate_provenance(rows):
    signatures = {}
    for row in rows:
        if row.get("semantic_cache_schema_version") not in {
            LEGACY_PREDICTION_SCHEMA_VERSION,
            SEMANTIC_CACHE_SCHEMA_VERSION,
        }:
            raise ValueError("unsupported semantic prediction schema")
        role = _prompt_role(row)
        prompt_version = row.get("prompt_version")
        if not isinstance(prompt_version, str) or not prompt_version:
            raise ValueError("invalid prompt_version")
        prompt_sha256 = row.get("prompt_sha256")
        if not isinstance(prompt_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", prompt_sha256
        ):
            raise ValueError("invalid prompt_sha256")
        model = row.get("model")
        wire_api = row.get("wire_api")
        if not isinstance(model, str) or not model or wire_api not in {
            "chat",
            "responses",
        }:
            raise ValueError("invalid VLM provenance")
        response_contract_version = row.get("response_contract_version")
        signature = {
            "prompt_version": prompt_version,
            "prompt_sha256": prompt_sha256,
            "response_contract_version": response_contract_version,
            "model": model,
            "wire_api": wire_api,
        }
        old = signatures.setdefault(role, signature)
        if old != signature:
            raise ValueError(
                "prediction file mixes protocol, model, or wire API within one role"
            )
    return dict(sorted(signatures.items()))


def _validate_prediction(prediction, manifest):
    required = {
        "candidate_id",
        "video_id",
        "absolute_feature_index",
        "absolute_center_time",
        "center_sec",
        "status",
        "prediction",
        "semantic_pass",
        "parse_error",
        "protocol_error",
        "consistency_error",
    }
    if not required.issubset(prediction):
        raise ValueError("prediction is missing a semantic protocol field")
    if prediction["status"] != "ok":
        raise ValueError("latest prediction row is not successful")
    if str(prediction["candidate_id"]) != str(manifest["candidate_id"]):
        raise ValueError("prediction candidate_id differs from manifest")
    if _prompt_role(prediction) != _prompt_role(manifest):
        raise ValueError("prediction prompt role differs from manifest")
    _absolute_time(prediction)
    if _semantic_key(prediction) != _semantic_key(manifest):
        raise ValueError("prediction exact semantic key differs from manifest")
    if any(
        prediction[field] is not False
        for field in ("parse_error", "protocol_error", "consistency_error")
    ):
        raise ValueError("prediction contains a protocol or parsing error")
    label = prediction["prediction"]
    if label not in VALID_PREDICTIONS:
        raise ValueError("unsupported semantic prediction label")
    semantic_pass = label == ACTION_PREDICTION
    if prediction["semantic_pass"] is not semantic_pass:
        raise ValueError("semantic_pass disagrees with VLM prediction")
    return label, semantic_pass, _prompt_role(manifest)


def build_entries(manifest_rows, prediction_rows):
    manifest_by_id = _validate_manifest(manifest_rows)
    _validate_provenance(prediction_rows)
    latest = {}
    for prediction in prediction_rows:
        candidate_id = str(prediction.get("candidate_id"))
        if candidate_id not in manifest_by_id:
            raise ValueError("prediction candidate is absent from manifest")
        latest[candidate_id] = prediction
    if set(latest) != set(manifest_by_id):
        raise ValueError("prediction coverage differs from manifest")

    entries = []
    for candidate_id, manifest in manifest_by_id.items():
        label, semantic_pass, role = _validate_prediction(
            latest[candidate_id], manifest
        )
        entries.append(
            {
                "prompt_role": role,
                "video_id": str(manifest["video_id"]),
                "absolute_feature_index": _absolute_index(manifest),
                "absolute_center_time": _absolute_time(manifest),
                "semantic_label": label,
                "semantic_pass": semantic_pass,
            }
        )
    return entries


def validate_selection_provenance(provenance):
    if not isinstance(provenance, dict) or set(provenance) != SELECTION_PROVENANCE_FIELDS:
        raise ValueError("semantic selection teacher provenance fields differ")
    for field in (
        "selection_teacher_checkpoint_sha256",
        "selection_teacher_state_sha256",
    ):
        if not isinstance(provenance[field], str) or not re.fullmatch(
            r"[0-9a-f]{64}", provenance[field]
        ):
            raise ValueError("invalid semantic selection teacher hash")
    if provenance["selection_teacher_weights"] not in {"student", "ema"}:
        raise ValueError("invalid semantic selection teacher weights")
    beta = float(provenance["selection_beta"])
    alpha = float(provenance["selection_alpha"])
    if not 0.0 <= beta < alpha <= 1.0:
        raise ValueError("invalid semantic selection thresholds")
    return {
        **provenance,
        "selection_beta": beta,
        "selection_alpha": alpha,
    }


def build_payload(manifest_rows, prediction_rows, selection_teacher_provenance):
    prompt_protocols = _validate_provenance(prediction_rows)
    entries = build_entries(manifest_rows, prediction_rows)
    counts = Counter(
        (entry["prompt_role"], entry["semantic_label"]) for entry in entries
    )
    return {
        "version": 7,
        "semantic_cache_schema_version": SEMANTIC_CACHE_SCHEMA_VERSION,
        "source": "offline_vlm_candidate_blind_bidirectional_semantics",
        "key_fields": ["prompt_role", "video_id", "absolute_feature_index"],
        "selection_teacher_provenance": validate_selection_provenance(
            selection_teacher_provenance
        ),
        "prompt_protocols": prompt_protocols,
        "counts": {
            role: {
                label: count
                for (entry_role, label), count in sorted(counts.items())
                if entry_role == role
            }
            for role in sorted({role for role, _ in counts})
        },
        "entries": entries,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest-metadata", default=None)
    args = parser.parse_args()
    metadata_path = Path(args.manifest_metadata or (args.manifest + ".meta.json"))
    if not metadata_path.is_file():
        raise ValueError("semantic manifest metadata not found: {}".format(metadata_path))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("selection_uses_target_gt") is not False:
        raise ValueError("semantic candidate selection must forbid target GT")
    payload = build_payload(
        read_jsonl(args.manifest),
        read_jsonl(args.predictions),
        metadata.get("selection_teacher_provenance"),
    )
    payload["manifest_sha256"] = hashlib.sha256(
        Path(args.manifest).read_bytes()
    ).hexdigest()
    payload["predictions_sha256"] = hashlib.sha256(
        Path(args.predictions).read_bytes()
    ).hexdigest()
    payload["manifest_metadata_sha256"] = hashlib.sha256(
        metadata_path.read_bytes()
    ).hexdigest()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), **payload["counts"]}, indent=2))


if __name__ == "__main__":
    main()

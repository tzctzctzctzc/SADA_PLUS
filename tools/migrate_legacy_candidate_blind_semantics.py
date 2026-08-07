"""Migrate the frozen v1 candidate-blind audit into the semantic-only cache."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from tools.build_vlm_training_cache import (
    SEMANTIC_CACHE_SCHEMA_VERSION,
    build_payload,
    read_jsonl,
)


LEGACY_ROUTING_SCHEMA = "candidate-blind-exact-anchor-routing-v1"
LEGACY_PROMPT_VERSION = "candidate-blind-target-foreground-gate-v1"
LABEL_MAPPING = {"TARGET_FOREGROUND": "TARGET_ACTION"}


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _absolute_feature_index(row):
    value = float(row["anchor_index"]) * float(row["point_stride"])
    rounded = int(round(value))
    if rounded < 0 or abs(value - rounded) > 1e-6:
        raise ValueError("legacy anchor does not map to an integer feature index")
    return rounded


def _same_number(left, right):
    left = float(left)
    right = float(right)
    return math.isfinite(left) and math.isfinite(right) and abs(left - right) <= 1e-6


def normalize_legacy_rows(manifest_rows, prediction_rows):
    manifests = {}
    semantic_keys = set()
    normalized_manifests = []
    for row in manifest_rows:
        candidate_id = str(row["candidate_id"])
        if candidate_id in manifests:
            raise ValueError("duplicate legacy candidate_id: {}".format(candidate_id))
        absolute_index = _absolute_feature_index(row)
        key = (str(row["video_id"]), absolute_index)
        if key in semantic_keys:
            raise ValueError("duplicate legacy absolute semantic key: {}".format(key))
        semantic_keys.add(key)
        manifests[candidate_id] = row
        normalized_manifests.append(
            {
                "candidate_id": candidate_id,
                "video_id": key[0],
                "absolute_feature_index": absolute_index,
                "absolute_center_time": float(row["absolute_center_time"]),
                "center_sec": float(row["center_sec"]),
            }
        )

    latest = {}
    signatures = set()
    for row in prediction_rows:
        candidate_id = str(row.get("candidate_id"))
        if candidate_id not in manifests:
            raise ValueError("legacy prediction is absent from manifest")
        latest[candidate_id] = row
        signatures.add(
            (
                row.get("routing_schema_version"),
                row.get("prompt_version"),
                row.get("prompt_sha256"),
                row.get("model"),
                row.get("wire_api"),
            )
        )
    if set(latest) != set(manifests):
        raise ValueError("legacy prediction coverage differs from manifest")
    if len(signatures) != 1:
        raise ValueError("legacy predictions mix protocol provenance")
    routing_schema, prompt_version, prompt_sha256, model, wire_api = signatures.pop()
    if routing_schema != LEGACY_ROUTING_SCHEMA:
        raise ValueError("unsupported legacy routing schema")
    if prompt_version != LEGACY_PROMPT_VERSION:
        raise ValueError("unsupported legacy prompt version")
    if not isinstance(prompt_sha256, str) or len(prompt_sha256) != 64:
        raise ValueError("invalid legacy prompt hash")

    normalized_predictions = []
    for manifest in normalized_manifests:
        candidate_id = manifest["candidate_id"]
        legacy_manifest = manifests[candidate_id]
        prediction = latest[candidate_id]
        if prediction.get("status") != "ok" or any(
            prediction.get(field) is not False
            for field in ("parse_error", "protocol_error", "consistency_error")
        ):
            raise ValueError("legacy prediction is not a clean successful response")
        if str(prediction.get("video_id")) != manifest["video_id"]:
            raise ValueError("legacy prediction video differs from manifest")
        for field in ("anchor_index", "fpn_level", "global_anchor_index"):
            if prediction.get(field) != legacy_manifest.get(field):
                raise ValueError("legacy prediction {} differs from manifest".format(field))
        if not _same_number(
            prediction.get("absolute_center_time"), manifest["absolute_center_time"]
        ) or not _same_number(prediction.get("center_sec"), manifest["center_sec"]):
            raise ValueError("legacy prediction time differs from manifest")

        old_label = prediction.get("prediction")
        label = LABEL_MAPPING.get(old_label, old_label)
        if label not in {
            "TARGET_ACTION",
            "OTHER_ACTIVITY",
            "TRUE_BACKGROUND",
            "UNCERTAIN",
        }:
            raise ValueError("unsupported legacy semantic label")
        normalized_predictions.append(
            {
                **manifest,
                "semantic_cache_schema_version": SEMANTIC_CACHE_SCHEMA_VERSION,
                "prompt_sha256": prompt_sha256,
                "model": model,
                "wire_api": wire_api,
                "status": "ok",
                "prediction": label,
                "semantic_pass": label == "TARGET_ACTION",
                "parse_error": False,
                "protocol_error": False,
                "consistency_error": False,
            }
        )
    return normalized_manifests, normalized_predictions


def migrate_payload(manifest_rows, prediction_rows, manifest_sha256, predictions_sha256):
    manifests, predictions = normalize_legacy_rows(manifest_rows, prediction_rows)
    payload = build_payload(manifests, predictions)
    payload["migration"] = {
        "source_routing_schema": LEGACY_ROUTING_SCHEMA,
        "source_prompt_version": LEGACY_PROMPT_VERSION,
        "input_manifest_sha256": manifest_sha256,
        "input_predictions_sha256": predictions_sha256,
        "label_mapping": LABEL_MAPPING,
    }
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = migrate_payload(
        read_jsonl(args.manifest),
        read_jsonl(args.predictions),
        file_sha256(args.manifest),
        file_sha256(args.predictions),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), **payload["counts"]}, indent=2))


if __name__ == "__main__":
    main()

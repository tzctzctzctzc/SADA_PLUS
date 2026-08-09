"""Migrate the frozen v1 candidate-blind audit into the semantic-only cache."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

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


def state_dict_sha256(state_dict):
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


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
                "prompt_role": "low_rescue",
                "prompt_version": prediction.get(
                    "prompt_version", LEGACY_PROMPT_VERSION
                ),
                "prompt_sha256": prompt_sha256,
                "response_contract_version": "legacy-candidate-blind-migration-v1",
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


def migrate_payload(
    manifest_rows,
    prediction_rows,
    manifest_sha256,
    predictions_sha256,
    selection_teacher_provenance,
):
    manifests, predictions = normalize_legacy_rows(manifest_rows, prediction_rows)
    payload = build_payload(manifests, predictions, selection_teacher_provenance)
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
    parser.add_argument("--manifest-metadata", default=None)
    args = parser.parse_args()
    metadata_path = Path(args.manifest_metadata or (args.manifest + ".meta.json"))
    if not metadata_path.is_file():
        raise ValueError("legacy manifest metadata not found: {}".format(metadata_path))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("selection_uses_target_gt") is not False:
        raise ValueError("legacy candidate selection must forbid target GT")
    checkpoint_path = Path(metadata["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    weights = metadata["weights"]
    state_key = "ema_state_dict" if weights == "ema" else "state_dict"
    if state_key not in checkpoint:
        raise ValueError("legacy selection checkpoint has no {}".format(state_key))
    selection_teacher_provenance = {
        "selection_teacher_checkpoint_sha256": file_sha256(checkpoint_path),
        "selection_teacher_state_sha256": state_dict_sha256(checkpoint[state_key]),
        "selection_teacher_weights": weights,
        "selection_beta": float(metadata["beta"]),
        "selection_alpha": float(metadata["alpha"]),
    }
    payload = migrate_payload(
        read_jsonl(args.manifest),
        read_jsonl(args.predictions),
        file_sha256(args.manifest),
        file_sha256(args.predictions),
        selection_teacher_provenance,
    )
    payload["manifest_metadata_sha256"] = file_sha256(metadata_path)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), **payload["counts"]}, indent=2))


if __name__ == "__main__":
    main()

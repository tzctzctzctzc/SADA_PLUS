import argparse
import json
import re
from pathlib import Path

from tools.calibrate_vlm_temperature import (
    row_class_logits,
    temperature_scaled_probabilities,
)


KNOWN_LABELS = (
    "TAKE", "PUT", "WASH", "OPEN", "CLOSE", "INSERT",
    "TURN_ON", "CUT", "TURN_OFF", "POUR",
)
VALID_LABELS = KNOWN_LABELS + ("UNKNOWN_ACTION", "BACKGROUND", "ABSTAIN")


def manifest_split(manifest_path, metadata):
    if metadata.get("split"):
        return metadata["split"]
    name_tokens = set(re.split(r"[^a-z0-9]+", Path(manifest_path).name.lower()))
    for split in ("train", "val", "test"):
        if split in name_tokens:
            return split
    raise ValueError("manifest metadata is missing split and filename cannot infer it")


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_cache(manifest_path, prediction_paths, temperature=None):
    manifest = read_jsonl(manifest_path)
    by_id = {row["candidate_id"]: row for row in manifest}
    if len(by_id) != len(manifest):
        raise ValueError("proposal manifest contains duplicate candidates")
    predictions = {}
    for path in prediction_paths:
        for row in read_jsonl(path):
            if row.get("status") != "ok":
                continue
            candidate_id = row["candidate_id"]
            if candidate_id in predictions:
                raise ValueError("duplicate successful VLM prediction")
            predictions[candidate_id] = row

    entries = []
    protocols = set()
    for candidate_id, prediction in sorted(predictions.items()):
        if candidate_id not in by_id:
            raise ValueError("VLM prediction is absent from the manifest")
        source = by_id[candidate_id]
        if (
            str(prediction["video_id"]) != str(source["video_id"])
            or int(prediction["absolute_feature_index"])
            != int(source["absolute_feature_index"])
        ):
            raise ValueError("VLM prediction identity differs from the manifest")
        label = prediction["prediction"]
        if label not in VALID_LABELS:
            raise ValueError("VLM prediction has an invalid semantic label")
        protocols.add((
            prediction["model"],
            prediction["prompt_version"],
            prediction.get("base_prompt_sha256", prediction["prompt_sha256"]),
            prediction.get("wire_api"),
        ))
        entry = {
            "candidate_id": candidate_id,
            "video_id": str(source["video_id"]),
            "absolute_feature_index": int(source["absolute_feature_index"]),
            "absolute_center_time": float(source["absolute_center_time"]),
            "proposal_start_sec": float(source["proposal_start_sec"]),
            "proposal_end_sec": float(source["proposal_end_sec"]),
            "source_fpn_levels": [
                int(level) for level in source["source_fpn_levels"]
            ],
            "semantic_label": label,
        }
        if label in KNOWN_LABELS:
            entry["class_id"] = KNOWN_LABELS.index(label)
            probabilities = prediction.get("class_probabilities")
            if temperature is not None:
                probabilities = temperature_scaled_probabilities(
                    row_class_logits(prediction), temperature
                )
            if probabilities is not None:
                probabilities = [float(value) for value in probabilities]
                if (
                    len(probabilities) != len(KNOWN_LABELS)
                    or any(value < 0.0 for value in probabilities)
                    or abs(sum(probabilities) - 1.0) > 1e-4
                ):
                    raise ValueError("invalid VLM class probabilities")
                entry["class_probabilities"] = probabilities
                entry["confidence"] = max(probabilities)
        entries.append(entry)
    if len(protocols) > 1:
        raise ValueError("VLM prediction shards use different prompt protocols")
    protocol = next(iter(protocols), (None, None, None, None))
    metadata_path = Path(str(manifest_path) + ".meta.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {
        "schema_version": "proposal-semantic-evidence-v1",
        "source": "offline_vlm_proposal_semantics",
        "domain": metadata["domain"],
        "split": manifest_split(manifest_path, metadata),
        "key_fields": ["candidate_id"],
        "selection_uses_gt": False,
        "selection_teacher_provenance": metadata[
            "selection_teacher_provenance"
        ],
        "prompt_protocol": {
            "model": protocol[0],
            "prompt_version": protocol[1],
            "prompt_sha256": protocol[2],
            "wire_api": protocol[3],
        },
        "manifest_candidates": len(manifest),
        "successful_predictions": len(entries),
        "entries": entries,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--temperature-calibration-report")
    args = parser.parse_args()
    temperature = None
    if args.temperature_calibration_report:
        calibration = json.loads(
            Path(args.temperature_calibration_report).read_text(encoding="utf-8")
        )
        if calibration.get("selection_uses_target_gt") is not False:
            raise ValueError("temperature calibration must be source-only")
        temperature = float(calibration["temperature"])
    payload = build_cache(args.manifest, args.predictions, temperature)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({
        "output": str(output),
        "manifest_candidates": payload["manifest_candidates"],
        "successful_predictions": payload["successful_predictions"],
    }, indent=2))


if __name__ == "__main__":
    main()

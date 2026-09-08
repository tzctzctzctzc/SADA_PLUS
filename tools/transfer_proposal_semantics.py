"""Transfer frozen VLM semantics onto proposals from another TAL teacher."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def temporal_iou(first, second):
    intersection = max(
        0.0,
        min(first[1], second[1]) - max(first[0], second[0]),
    )
    union = (
        first[1] - first[0] + second[1] - second[0] - intersection
    )
    return intersection / max(union, 1e-8)


def transfer_semantics(
    manifest,
    source_payload,
    manifest_metadata,
    min_iou,
    min_confidence=0.0,
):
    if source_payload.get("target_gt_usage") not in (None, "forbidden"):
        raise ValueError("semantic source must not use target GT")
    by_video = defaultdict(list)
    for entry in source_payload["entries"]:
        probabilities = entry.get("class_probabilities")
        confidence = max(probabilities) if probabilities else 1.0
        if confidence < float(min_confidence):
            continue
        by_video[str(entry["video_id"])].append(entry)

    entries = []
    for proposal in manifest:
        interval = (
            float(proposal["proposal_start_sec"]),
            float(proposal["proposal_end_sec"]),
        )
        candidates = [
            (temporal_iou(interval, (
                float(entry["proposal_start_sec"]),
                float(entry["proposal_end_sec"]),
            )), entry)
            for entry in by_video.get(str(proposal["video_id"]), [])
        ]
        if not candidates:
            continue
        overlap, semantic = max(candidates, key=lambda item: item[0])
        if overlap < float(min_iou):
            continue
        entry = {
            "candidate_id": str(proposal["candidate_id"]),
            "video_id": str(proposal["video_id"]),
            "absolute_feature_index": int(
                proposal["absolute_feature_index"]
            ),
            "absolute_center_time": float(
                proposal["absolute_center_time"]
            ),
            "proposal_start_sec": interval[0],
            "proposal_end_sec": interval[1],
            "source_fpn_levels": [
                int(level) for level in proposal["source_fpn_levels"]
            ],
            "semantic_label": str(semantic["semantic_label"]),
            "semantic_source_candidate_id": str(semantic["candidate_id"]),
            "semantic_transfer_iou": float(overlap),
        }
        if semantic.get("class_probabilities") is not None:
            entry["class_probabilities"] = [
                float(value) for value in semantic["class_probabilities"]
            ]
        if "teacher_score" in proposal:
            entry["teacher_score"] = float(proposal["teacher_score"])
        entries.append(entry)

    return {
        "schema_version": "proposal-semantic-evidence-v1",
        "source": "offline_vlm_proposal_semantics",
        "domain": manifest_metadata["domain"],
        "split": manifest_metadata.get("split", source_payload.get("split")),
        "key_fields": ["candidate_id"],
        "selection_uses_gt": False,
        "target_gt_usage": "forbidden",
        "selection_teacher_provenance": manifest_metadata[
            "selection_teacher_provenance"
        ],
        "prompt_protocol": source_payload["prompt_protocol"],
        "manifest_candidates": len(manifest),
        "successful_predictions": len(entries),
        "selection_filter": {
            "semantic_transfer": "maximum_temporal_iou",
            "minimum_temporal_iou": float(min_iou),
            "minimum_semantic_confidence": float(min_confidence),
            "uses_target_gt": False,
        },
        "entries": entries,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--semantic-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-iou", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    args = parser.parse_args()
    if not (
        0.0 <= args.min_iou <= 1.0
        and 0.0 <= args.min_confidence <= 1.0
    ):
        raise ValueError("minimum IoU and confidence must be in [0, 1]")

    manifest_path = Path(args.manifest)
    semantic_path = Path(args.semantic_cache)
    manifest_metadata = json.loads(
        Path(str(manifest_path) + ".meta.json").read_text(encoding="utf-8")
    )
    source_payload = json.loads(semantic_path.read_text(encoding="utf-8"))
    payload = transfer_semantics(
        read_jsonl(manifest_path),
        source_payload,
        manifest_metadata,
        args.min_iou,
        args.min_confidence,
    )
    payload["semantic_source_sha256"] = hashlib.sha256(
        semantic_path.read_bytes()
    ).hexdigest()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({
        "output": str(output),
        "manifest_candidates": len(read_jsonl(manifest_path)),
        "transferred_entries": len(payload["entries"]),
        "minimum_temporal_iou": args.min_iou,
        "minimum_semantic_confidence": args.min_confidence,
    }, indent=2))


if __name__ == "__main__":
    main()

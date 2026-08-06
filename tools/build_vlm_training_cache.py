"""Convert listwise VLM audit outputs into a GT-free training cache."""

import argparse
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_entries(manifest_rows, prediction_rows):
    predictions = {
        row["candidate_id"]: row
        for row in prediction_rows
        if row.get("status") == "ok"
    }
    entries = []
    for row in manifest_rows:
        prediction = predictions.get(row["candidate_id"])
        if prediction is None:
            continue
        label = prediction["prediction"]
        selected = row["option_mapping"].get(label)
        if selected and selected["role"] == "top1":
            decision = "action"
            class_id = int(row["sada_top1_class"])
        elif label == "TRUE_BACKGROUND":
            decision = "background"
            class_id = None
        else:
            decision = "ignore"
            class_id = None
        members = row.get("member_anchors") or [
            {
                "fpn_level": row["fpn_level"],
                "anchor_index": row["anchor_index"],
            }
        ]
        for member in members:
            entry = {
                "video_id": row["video_id"],
                "fpn_level": int(member["fpn_level"]),
                "anchor_index": int(member["anchor_index"]),
                "decision": decision,
            }
            if class_id is not None:
                entry["class_id"] = class_id
            entries.append(entry)
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    entries = build_entries(
        read_jsonl(args.manifest), read_jsonl(args.predictions)
    )
    payload = {
        "version": 1,
        "source": "offline_listwise_vlm_without_target_gt",
        "routing_rule": {
            "top1_selected": "action",
            "true_background": "background",
            "all_other_outputs": "ignore",
        },
        "counts": dict(Counter(entry["decision"] for entry in entries)),
        "entries": entries,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), **payload["counts"]}, indent=2))


if __name__ == "__main__":
    main()

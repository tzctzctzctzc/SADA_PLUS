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


EXACT_KEY_FIELDS = (
    "video_id",
    "fpn_level",
    "global_anchor_index",
    "absolute_center_time",
    "teacher_top1_class",
)


def build_entries(manifest_rows, prediction_rows, routing_policy="legacy_listwise"):
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
        if routing_policy == "exact_anchor_rescue":
            if any(field not in row for field in EXACT_KEY_FIELDS):
                raise ValueError("exact-anchor manifest is missing a strict key field")
            members = row.get("member_anchors") or []
            if len(members) != 1:
                raise ValueError("exact-anchor query must have exactly one member")
            member = members[0]
            if (
                int(member["fpn_level"]) != int(row["fpn_level"])
                or int(member["anchor_index"]) != int(row["anchor_index"])
                or int(member["global_anchor_index"])
                != int(row["global_anchor_index"])
                or float(member["absolute_center_time"])
                != float(row["absolute_center_time"])
                or int(member["teacher_top1_class"])
                != int(row["teacher_top1_class"])
            ):
                raise ValueError("exact-anchor member does not match its query key")
            decision = (
                "action"
                if label == "CONFIRM_KEY_ACTION_EVIDENCE"
                else "ignore"
            )
            class_id = int(row["teacher_top1_class"]) if decision == "action" else None
        else:
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
            if routing_policy == "exact_anchor_rescue":
                entry.update(
                    {
                        "global_anchor_index": int(row["global_anchor_index"]),
                        "absolute_center_time": float(row["absolute_center_time"]),
                        "teacher_top1_class": int(row["teacher_top1_class"]),
                    }
                )
            if class_id is not None:
                entry["class_id"] = class_id
            entries.append(entry)
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--routing-policy",
        choices=("legacy_listwise", "exact_anchor_rescue"),
        default="legacy_listwise",
    )
    args = parser.parse_args()

    entries = build_entries(
        read_jsonl(args.manifest),
        read_jsonl(args.predictions),
        routing_policy=args.routing_policy,
    )
    payload = {
        "version": 2 if args.routing_policy == "exact_anchor_rescue" else 1,
        "source": "offline_exact_anchor_vlm_without_target_gt"
        if args.routing_policy == "exact_anchor_rescue"
        else "offline_listwise_vlm_without_target_gt",
        "routing_policy": args.routing_policy,
        "routing_rule": {
            "confirm_key_action_evidence": "action",
            "all_non_confirm_outputs": "ignore",
        }
        if args.routing_policy == "exact_anchor_rescue"
        else {
            "top1_selected": "action",
            "true_background": "background",
            "all_other_outputs": "ignore",
        },
        "bank_key_fields": list(EXACT_KEY_FIELDS)
        if args.routing_policy == "exact_anchor_rescue"
        else None,
        "counts": dict(Counter(entry["decision"] for entry in entries)),
        "entries": entries,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), **payload["counts"]}, indent=2))


if __name__ == "__main__":
    main()

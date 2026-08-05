"""Build a position-balanced three-candidate listwise audit manifest."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.build_vlm_candidate_selectivity_manifest import (
    NEAR_CONFUSION,
    load_names,
    read_jsonl,
)


PERMUTATIONS = (
    ("top1", "top2", "near_confusion"),
    ("top1", "near_confusion", "top2"),
    ("top2", "top1", "near_confusion"),
    ("top2", "near_confusion", "top1"),
    ("near_confusion", "top1", "top2"),
    ("near_confusion", "top2", "top1"),
)


def stable_order(anchors, seed):
    return sorted(
        anchors,
        key=lambda row: hashlib.sha256(
            "{}:{}".format(seed, row["candidate_id"]).encode()
        ).hexdigest(),
    )


def build_rows(anchors, names, seed):
    if len(anchors) % len(PERMUTATIONS):
        raise ValueError("anchor count must be divisible by six for exact balance")
    rows = []
    for index, anchor in enumerate(stable_order(anchors, seed)):
        top1 = int(anchor["sada_top1_class"])
        candidates = {
            "top1": top1,
            "top2": int(anchor["sada_top2_class"]),
            "near_confusion": NEAR_CONFUSION[top1],
        }
        if len(set(candidates.values())) != 3:
            raise ValueError("listwise candidates are not unique for {}".format(
                anchor["candidate_id"]
            ))
        permutation = PERMUTATIONS[index % len(PERMUTATIONS)]
        option_mapping = {
            letter: {
                "class_id": candidates[role],
                "class_name": names[candidates[role]],
                "role": role,
            }
            for letter, role in zip("ABC", permutation)
        }
        rows.append(
            {
                **anchor,
                "anchor_id": anchor["candidate_id"],
                "frame_cache_id": anchor["candidate_id"],
                "option_mapping": option_mapping,
                "permutation_id": index % len(PERMUTATIONS),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--ontology-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    anchors = read_jsonl(args.manifest)
    required = {"sada_top2_class", "center_all_gt_class_ids"}
    if any(not required.issubset(row) for row in anchors):
        raise ValueError("input manifest must contain top-2 and all-class GT")
    names = load_names(args.ontology_csv)
    rows = build_rows(anchors, names, args.seed)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    position_counts = Counter()
    permutation_counts = Counter()
    for row in rows:
        permutation_counts[row["permutation_id"]] += 1
        for letter, option in row["option_mapping"].items():
            position_counts[(option["role"], letter)] += 1
    metadata = {
        "audit": "listwise_candidate_competition",
        "valid_uda_result": False,
        "anchor_selection": "reused exact frozen SADA anchors",
        "target_gt_use": "post-inference scoring only",
        "candidate_roles": ["top1", "top2", "near_confusion"],
        "position_assignment": "six permutations with exact balance",
        "seed": args.seed,
        "anchors": len(rows),
        "candidate_queries": len(rows),
        "permutation_counts": dict(sorted(permutation_counts.items())),
        "role_position_counts": {
            "{}@{}".format(role, letter): count
            for (role, letter), count in sorted(position_counts.items())
        },
        "near_confusion_mapping": {
            names[source]: names[target]
            for source, target in NEAR_CONFUSION.items()
        },
    }
    Path(str(output) + ".meta.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

"""Expand fixed SADA anchors into four candidate-selectivity controls."""

import argparse
import csv
import json
import random
from pathlib import Path


NEAR_CONFUSION = {
    0: 12,  # take vs remove
    1: 11,  # put vs move
    2: 29,  # wash vs scrub
    3: 31,  # open vs pull
    4: 36,  # close vs wrap/seal
    5: 26,  # insert vs fill
    6: 17,  # turn-on vs adjust
    7: 25,  # cut vs scrape
    8: 17,  # turn-off vs adjust
    9: 20,  # pour vs empty
}


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_names(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return {int(row["id"]): row["key"] for row in csv.DictReader(stream)}


def build_rows(anchors, names, seed):
    output = []
    for anchor in anchors:
        anchor_id = anchor["candidate_id"]
        top1 = int(anchor["sada_top1_class"])
        top2 = int(anchor["sada_top2_class"])
        near = NEAR_CONFUSION[top1]
        all_gt = {int(value) for value in anchor["center_all_gt_class_ids"]}
        random_pool = [
            class_id
            for class_id in range(10)
            if class_id not in {top1, top2} and class_id not in all_gt
        ]
        random_class = random.Random("{}:{}".format(seed, anchor_id)).choice(
            random_pool
        )
        candidates = (
            ("top1", top1),
            ("top2", top2),
            ("near_confusion", near),
            ("random_wrong", random_class),
        )
        if len({class_id for _, class_id in candidates}) != 4:
            raise ValueError("candidate roles are not unique for {}".format(anchor_id))
        for role, class_id in candidates:
            output.append(
                {
                    **anchor,
                    "candidate_id": "{}--{}--c{:02d}".format(
                        anchor_id, role, class_id
                    ),
                    "anchor_id": anchor_id,
                    "frame_cache_id": anchor_id,
                    "candidate_role": role,
                    "candidate_label_id": class_id,
                    "candidate_name": names[class_id],
                    "candidate_is_center_gt": class_id in all_gt,
                }
            )
    return output


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
        raise ValueError("input manifest must be enriched with top-2 and all-class GT")
    names = load_names(args.ontology_csv)
    rows = build_rows(anchors, names, args.seed)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    metadata = {
        "audit": "candidate_selectivity",
        "valid_uda_result": False,
        "anchor_selection": "reused exact frozen SADA anchors",
        "candidate_roles": ["top1", "top2", "near_confusion", "random_wrong"],
        "random_wrong_uses_gt": True,
        "target_gt_use": "construct random negative control and score only",
        "seed": args.seed,
        "anchors": len(anchors),
        "candidate_queries": len(rows),
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

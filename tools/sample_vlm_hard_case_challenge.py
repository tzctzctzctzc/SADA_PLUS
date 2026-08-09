"""Build a deterministic GT-enriched VLM capability-audit subset.

Target GT is used only to stratify this diagnostic challenge set. It must not
be used for training, threshold selection, or deployment routing.
"""

import argparse
import collections
import json
import random
from pathlib import Path


GROUPS = (
    "low_valid_correct",
    "low_semantic_invalid",
    "low_background",
    "high_background",
    "high_correct_control",
)


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def classify(row):
    band = row.get("score_band")
    gt_valid = bool(row["gt_is_foreground"])
    center_action = bool(row["center_in_target_action"])
    teacher_correct = bool(row["sada_top1_is_correct"])
    if band == "low" and gt_valid and teacher_correct:
        return "low_valid_correct"
    if band == "low" and not gt_valid and center_action:
        return "low_semantic_invalid"
    if band == "low" and not gt_valid and not center_action:
        return "low_background"
    if band == "high" and not gt_valid and not center_action:
        return "high_background"
    if band == "high" and gt_valid and teacher_correct:
        return "high_correct_control"
    return None


def balanced_sample(rows, count, seed):
    """Round-robin over video×level cells to avoid a few videos dominating."""
    rng = random.Random(seed)
    cells = collections.defaultdict(list)
    for row in rows:
        cells[(row["video_id"], int(row["fpn_level"]))].append(row)
    keys = list(cells)
    rng.shuffle(keys)
    for key in keys:
        rng.shuffle(cells[key])

    selected = []
    while keys and len(selected) < count:
        next_keys = []
        for key in keys:
            selected.append(cells[key].pop())
            if cells[key]:
                next_keys.append(key)
            if len(selected) == count:
                break
        keys = next_keys
    return selected


def interleave(groups):
    rows = []
    index = 0
    while True:
        added = False
        for name in GROUPS:
            if index < len(groups[name]):
                rows.append(groups[name][index])
                added = True
        if not added:
            return rows
        index += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--low-manifest", required=True)
    parser.add_argument("--high-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--low-valid-correct", type=int, default=100)
    parser.add_argument("--low-semantic-invalid", type=int, default=50)
    parser.add_argument("--low-background", type=int, default=50)
    parser.add_argument("--high-background", type=int, default=100)
    parser.add_argument("--high-correct-control", type=int, default=100)
    args = parser.parse_args()

    requested = {
        name: int(getattr(args, name))
        for name in GROUPS
    }
    pools = {name: [] for name in GROUPS}
    for row in read_jsonl(args.low_manifest) + read_jsonl(args.high_manifest):
        group = classify(row)
        if group:
            pools[group].append(row)

    selected = {}
    for offset, name in enumerate(GROUPS):
        chosen = balanced_sample(
            pools[name],
            requested[name],
            args.seed + offset,
        )
        selected[name] = []
        for row in chosen:
            original_id = row["candidate_id"]
            selected[name].append(
                {
                    **row,
                    "anchor_id": original_id,
                    "candidate_id": "{}-{}-{}".format(
                        args.scenario, name, original_id
                    ),
                    "scenario": args.scenario,
                    "challenge_group": name,
                    "selection_uses_target_gt": True,
                    "target_gt_use": "challenge-set stratification only",
                }
            )

    rows = interleave(selected)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "scenario": args.scenario,
        "seed": args.seed,
        "selection_uses_target_gt": True,
        "target_gt_use": "challenge-set stratification only",
        "population_estimate": False,
        "sampling": "balanced round-robin over video x FPN level cells",
        "requested": requested,
        "available": {name: len(pools[name]) for name in GROUPS},
        "selected": {name: len(selected[name]) for name in GROUPS},
        "total": len(rows),
    }
    Path(str(output) + ".meta.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

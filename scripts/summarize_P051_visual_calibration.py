#!/usr/bin/env python3
"""只诊断双规格与选项位置稳定性，不用结果调筛选阈值。"""
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=96)
    args = parser.parse_args()
    run = Path("runs/P051_tracebind")
    report, paired = {}, {}
    for model in ["large", "small"]:
        rows = list(map(json.loads, (run / f"visual_{model}_{args.limit}_mirror.jsonl").open()))
        assert len(rows) == args.limit * 4
        groups = defaultdict(list)
        for row in rows:
            groups[(row["bundle_id"], row["kind"])].append(row)
        correct, stable, biased = Counter(), Counter(), Counter()
        choices = {}
        for (bid, kind), values in groups.items():
            values.sort(key=lambda r: r["mirrored"])
            assert len(values) == 2 and [v["mirrored"] for v in values] == [0, 1]
            physical = [v["product_ids"][1 if v["prediction"] == "A" else 2] if v["prediction"] in ["A", "B"] else v["prediction"] for v in values]
            family = values[0]["family"]
            stable[family] += int(physical[0] == physical[1])
            biased[family] += int(values[0]["prediction"] == values[1]["prediction"] and values[0]["prediction"] in ["A", "B"])
            correct[family] += int(all(v["pass"] for v in values))
            choices[(bid, kind)] = physical
        counts = Counter(values[0]["family"] for values in groups.values())
        report[model] = {"comparisons": len(groups), "family_comparisons": dict(counts), "position_consistent": dict(stable),
                         "same_letter_after_swap": dict(biased), "metadata_relative_label_agreement_with_confidence": dict(correct),
                         "prediction_counts": dict(Counter(v["prediction"] for v in rows))}
        paired[model] = choices
    common = paired["large"].keys() & paired["small"].keys()
    report["cross_size_physical_decision_agreement"] = sum(paired["large"][key] == paired["small"][key] for key in common) / len(common)
    report["same_family_models_do_not_certify_ground_truth"] = True
    report["thresholds_adjusted_using_calibration"] = False
    out = run / "visual_calibration_audit.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Analyze fixed source-bound predicate conjunction against direct dual scoring."""
import ast
import collections
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.stats import rankdata

ROOT = Path("/vepfs/vlms/1")
assert Path.cwd().resolve() == ROOT
cfg_path = Path("configs/P035_source_bound_predicates.json")
cfg = json.loads(cfg_path.read_text())
for path, expected in cfg["source_sha256"].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected

cohort = json.loads(Path(cfg["cohort_config"]).read_text())
rows = cohort["records"]
source_predictions = Path("runs/P035_source_bound_predicates/predictions.jsonl")
predictions = [json.loads(line) for line in source_predictions.read_text().splitlines()]
assert len(predictions) == len({row["key"] for row in predictions}) == cfg["expected_records"]
assert all(math.isfinite(row["margin"]) for row in predictions)
lookup = {
    (row["id"], row["condition"], row["candidate"]): float(row["margin"])
    for row in predictions
}
p030_predictions = [
    json.loads(line)
    for line in Path("runs/P030_pinpoint_references/predictions.jsonl").read_text().splitlines()
]
dual_lookup = {
    (row["id"], row["candidate"]): float(row["margin"])
    for row in p030_predictions
    if row["condition"] == "labelled_dual"
}

metric_source = Path("third_party/pinpoint-dataset/utils/metrics.py")
manifest = json.loads(Path("literature/PinPoint_source_manifest.json").read_text())
assert hashlib.sha256(metric_source.read_bytes()).hexdigest() == next(
    row["sha256"] for row in manifest["files"] if row["path"] == "utils/metrics.py"
)
namespace = {"Dict": Dict, "List": List}
exec(
    compile(
        ast.Module(
            body=[
                node
                for node in ast.parse(metric_source.read_text()).body
                if isinstance(node, ast.FunctionDef)
            ],
            type_ignores=[],
        ),
        str(metric_source),
        "exec",
    ),
    namespace,
)


def metrics(rank, row):
    official = namespace["calculate_metrics_at_k"](rank, row["positive_set"], 10)
    return {
        "ap@10": float(official["ap"]),
        "precision@1": float(rank[0] in row["positive_set"]),
        "precision@10": float(sum(x in row["positive_set"] for x in rank[:10]) / 10),
        "negative_precision@10": float(sum(x in row["negative_set"] for x in rank[:10]) / 10),
    }


def auc(positive, negative):
    if not positive or not negative:
        return None
    scores = np.asarray(positive + negative, dtype=np.float64)
    labels = np.asarray([1] * len(positive) + [0] * len(negative))
    ranks = rankdata(scores, method="average")
    return float(
        (ranks[labels == 1].sum() - len(positive) * (len(positive) + 1) / 2)
        / (len(positive) * len(negative))
    )


images = {
    row["signature"]: row["pixel_sha256"]
    for row in json.loads(Path("data/PinPoint/P030_image_manifest.json").read_text())["entries"]
}
parent = list(range(len(rows)))


def find(index):
    if parent[index] != index:
        parent[index] = find(parent[index])
    return parent[index]


owner = {}
for index, row in enumerate(rows):
    for signature in row["references"]:
        pixel_hash = images[signature]
        if pixel_hash in owner:
            parent[find(index)] = find(owner[pixel_hash])
        else:
            owner[pixel_hash] = index
groups = collections.defaultdict(list)
for index in range(len(rows)):
    groups[find(index)].append(index)
blocks = list(groups.values())
rng = np.random.default_rng(cfg["bootstrap"]["seed"])
bootstrap = [
    np.concatenate([blocks[index] for index in rng.integers(len(blocks), size=len(blocks))])
    for _ in range(cfg["bootstrap"]["replicates"])
]

names = ["dual_raw", "source1", "source2", "log_prob_and", "min_margin", "sum_margin"]
values = {name: [] for name in names}
macro_auc = {name: [] for name in names}
ties = collections.Counter()
details = []

for row in rows:
    candidates = row["candidates_to_score"]
    source1 = np.asarray(
        [lookup[row["id"], "source1", candidate] for candidate in candidates],
        dtype=np.float64,
    )
    source2 = np.asarray(
        [lookup[row["id"], "source2", candidate] for candidate in candidates],
        dtype=np.float64,
    )
    dual = np.asarray(
        [dual_lookup[row["id"], candidate] for candidate in candidates],
        dtype=np.float64,
    )
    scores = {
        "dual_raw": dual,
        "source1": source1,
        "source2": source2,
        "log_prob_and": -np.logaddexp(0, -source1) - np.logaddexp(0, -source2),
        "min_margin": np.minimum(source1, source2),
        "sum_margin": source1 + source2,
    }
    row_result = {
        "id": row["id"],
        "reference_component": row["reference_component"],
        "metrics": {},
        "scores": {},
        "ranks": {},
        "mechanism_auc": {},
    }
    for name, score in scores.items():
        order = np.argsort(-score, kind="stable")
        rank = [candidates[index] for index in order] + row["original_candidates"][20:]
        assert len(rank) == len(set(rank)) == 50
        result = metrics(rank, row)
        values[name].append(result)
        row_result["metrics"][name] = result
        row_result["scores"][name] = score.tolist()
        row_result["ranks"][name] = rank
        ties[name] += int(len(set(score.tolist())) < len(score))
        positive = [
            float(value)
            for candidate, value in zip(candidates, score)
            if candidate in row["positive_set"]
        ]
        negative = [
            float(value)
            for candidate, value in zip(candidates, score)
            if candidate in row["negative_set"]
        ]
        macro_auc[name].append(auc(positive, negative))
        row_result["mechanism_auc"][name] = macro_auc[name][-1]
    details.append(row_result)

summaries = {
    name: {
        metric: float(np.mean([result[metric] for result in results]))
        for metric in results[0]
    }
    for name, results in values.items()
}
paired = {}
for method in ["log_prob_and", "min_margin", "sum_margin", "source1", "source2"]:
    comparison = method + " minus dual_raw"
    paired[comparison] = {}
    for metric in ["ap@10", "precision@1", "precision@10", "negative_precision@10"]:
        delta = np.asarray([x[metric] for x in values[method]]) - np.asarray(
            [x[metric] for x in values["dual_raw"]]
        )
        paired[comparison][metric] = {
            "difference_pp": float(delta.mean() * 100),
            "reference_cluster_95ci_pp": np.percentile(
                [delta[index].mean() * 100 for index in bootstrap], [2.5, 97.5]
            ).tolist(),
            "query_gains": int((delta > 0).sum()),
            "query_losses": int((delta < 0).sum()),
        }

mechanism = {}
for name in names:
    eligible = [value for value in macro_auc[name] if value is not None]
    boot = []
    for indices in bootstrap:
        selected = [macro_auc[name][index] for index in indices if macro_auc[name][index] is not None]
        boot.append(float(np.mean(selected)) if selected else float("nan"))
    mechanism[name] = {
        "n_eligible_queries": len(eligible),
        "macro_query_auc_positive_vs_explicit_negative": float(np.mean(eligible)),
        "reference_cluster_95ci": np.nanpercentile(boot, [2.5, 97.5]).tolist(),
    }

primary = paired["log_prob_and minus dual_raw"]
gate_checks = {
    "and_minus_dual_ap10_gt_0": primary["ap@10"]["difference_pp"] > 0,
    "and_macro_auc_minus_dual_gt_0": mechanism["log_prob_and"]["macro_query_auc_positive_vs_explicit_negative"]
    > mechanism["dual_raw"]["macro_query_auc_positive_vs_explicit_negative"],
    "and_negative_precision_at10_minus_dual_le_0": primary["negative_precision@10"]["difference_pp"] <= 0,
}
report = {
    "complete": True,
    "evidence_status": cfg["evidence_status"],
    "n_queries": len(rows),
    "n_reference_rgb_components": len(blocks),
    "summaries": summaries,
    "paired": paired,
    "mechanism": mechanism,
    "ties_anywhere_in_top20_scores": dict(ties),
    "continuation_gate_evaluation": {
        "checks": gate_checks,
        "all_pass": all(gate_checks.values()),
        "meaning": cfg["continuation_gate"]["meaning"],
    },
    "score_file_sha256": hashlib.sha256(source_predictions.read_bytes()).hexdigest(),
    "config_sha256": hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
    "limits": cfg["limits"],
}
Path("runs/P035_source_bound_predicate_details.json").write_text(
    json.dumps(details, indent=2) + chr(10)
)
Path("results/P035_source_bound_predicates.json").write_text(
    json.dumps(report, indent=2) + chr(10)
)
for name in names:
    print(
        name,
        "AP10",
        round(summaries[name]["ap@10"], 6),
        "P1",
        round(summaries[name]["precision@1"], 6),
        "negP10",
        round(summaries[name]["negative_precision@10"], 6),
        "AUC",
        round(mechanism[name]["macro_query_auc_positive_vs_explicit_negative"], 6),
    )
print("PRIMARY", primary)
print("GATE", report["continuation_gate_evaluation"])

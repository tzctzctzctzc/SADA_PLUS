"""Fixed-alpha exploratory reference-necessity contrast on the P030 development cohort."""
import ast
import collections
import hashlib
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import rankdata

ROOT = Path("/vepfs/vlms/1")
assert Path.cwd().resolve() == ROOT

cfg_path = Path("configs/P034_reference_necessity.json")
cfg = json.loads(cfg_path.read_text())
for path, expected in cfg["source_sha256"].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path

cohort = json.loads(Path("configs/P030_pinpoint_references.json").read_text())
rows = cohort["records"]
predictions_path = Path("runs/P030_pinpoint_references/predictions.jsonl")
predictions = [json.loads(line) for line in predictions_path.read_text().splitlines()]
assert len(predictions) == len({row["key"] for row in predictions}) == 3840
point_lookup = {
    (row["id"], row["condition"], row["candidate"]): float(row["margin"])
    for row in predictions
}

metric_source = Path("third_party/pinpoint-dataset/utils/metrics.py")
manifest = json.loads(Path("literature/PinPoint_source_manifest.json").read_text())
expected_metric_sha = next(
    row["sha256"] for row in manifest["files"] if row["path"] == "utils/metrics.py"
)
assert hashlib.sha256(metric_source.read_bytes()).hexdigest() == expected_metric_sha
namespace = {"Dict": Dict, "List": List}
nodes = [
    node
    for node in ast.parse(metric_source.read_text()).body
    if isinstance(node, ast.FunctionDef)
]
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(metric_source), "exec"), namespace)


def metrics(rank, row):
    result = namespace["calculate_metrics_at_k"](rank, row["positive_set"], 10)
    return {
        "ap@10": float(result["ap"]),
        "precision@1": float(rank[0] in row["positive_set"]),
        "precision@10": float(sum(x in row["positive_set"] for x in rank[:10]) / 10),
        "negative_precision@10": float(sum(x in row["negative_set"] for x in rank[:10]) / 10),
    }


def auc(positive, negative):
    if not positive or not negative:
        return None
    values = np.asarray(positive + negative, dtype=np.float64)
    labels = np.asarray([1] * len(positive) + [0] * len(negative))
    ranks = rankdata(values, method="average")
    n_positive = len(positive)
    return float(
        (ranks[labels == 1].sum() - n_positive * (n_positive + 1) / 2)
        / (n_positive * len(negative))
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


def pointwise_scores(row):
    candidates = row["candidates_to_score"]
    dual = np.asarray(
        [point_lookup[row["id"], "labelled_dual", candidate] for candidate in candidates],
        dtype=np.float64,
    )
    single = np.asarray(
        [point_lookup[row["id"], "labelled_single", candidate] for candidate in candidates],
        dtype=np.float64,
    )
    return dual, single


embedding_cache = {}


def embedding_scores(model, row):
    folder = Path("runs/P031_coral_transfer") / model
    candidate_vectors = []
    for candidate in row["candidates_to_score"]:
        key = (model, candidate)
        if key not in embedding_cache:
            embedding_cache[key] = np.load(folder / ("candidate_" + candidate + ".npy"))
        candidate_vectors.append(embedding_cache[key])
    candidates = torch.tensor(np.stack(candidate_vectors), dtype=torch.float32)
    candidates = F.normalize(candidates, dim=-1)
    dual = torch.tensor(
        np.load(folder / ("query_" + row["id"] + "_dual.npy")), dtype=torch.float32
    )
    single = torch.tensor(
        np.load(folder / ("query_" + row["id"] + "_single.npy")), dtype=torch.float32
    )
    dual = (candidates @ F.normalize(dual, dim=-1)).numpy().astype(np.float64)
    single = (candidates @ F.normalize(single, dim=-1)).numpy().astype(np.float64)
    return dual, single


backends = {
    "qwen7b_pointwise": pointwise_scores,
    "base3b_embedding": lambda row: embedding_scores("base3b", row),
    "coral_embedding": lambda row: embedding_scores("coral", row),
}
details = []
summaries = {}
paired = {}
mechanisms = {}
ties = {}

for backend, score_function in backends.items():
    values = {name: [] for name in ["dual", "single", "rncr_alpha1", "source_lift"]}
    per_query_auc = []
    per_query_positive_lifts = []
    per_query_negative_lifts = []
    all_lifts = {"positive": [], "explicit_negative": [], "unjudged": []}
    ties[backend] = collections.Counter()

    for row in rows:
        dual, single = score_function(row)
        source_lift = dual - single
        scores = {
            "dual": dual,
            "single": single,
            "rncr_alpha1": 2 * dual - single,
            "source_lift": source_lift,
        }
        row_result = {
            "id": row["id"],
            "reference_component": row["reference_component"],
            "backend": backend,
            "metrics": {},
            "scores": {},
            "ranks": {},
        }
        for name, score in scores.items():
            order = np.argsort(-score, kind="stable")
            rank = [row["candidates_to_score"][index] for index in order]
            rank += row["original_candidates"][20:]
            assert len(rank) == len(set(rank)) == 50
            result = metrics(rank, row)
            values[name].append(result)
            row_result["metrics"][name] = result
            row_result["scores"][name] = score.tolist()
            row_result["ranks"][name] = rank
            ties[backend][name] += int(len(set(score.tolist())) < len(score))

        positive = []
        negative = []
        for candidate, lift in zip(row["candidates_to_score"], source_lift):
            value = float(lift)
            if candidate in row["positive_set"]:
                label = "positive"
                positive.append(value)
            elif candidate in row["negative_set"]:
                label = "explicit_negative"
                negative.append(value)
            else:
                label = "unjudged"
            all_lifts[label].append(value)
        query_auc = auc(positive, negative)
        per_query_auc.append(query_auc)
        per_query_positive_lifts.append(float(np.mean(positive)) if positive else None)
        per_query_negative_lifts.append(float(np.mean(negative)) if negative else None)
        row_result["mechanism_auc"] = query_auc
        details.append(row_result)

    summaries[backend] = {
        name: {
            metric: float(np.mean([result[metric] for result in results]))
            for metric in results[0]
        }
        for name, results in values.items()
    }
    paired[backend] = {}
    for method in ["rncr_alpha1", "source_lift", "single"]:
        comparison = method + " minus dual"
        paired[backend][comparison] = {}
        for metric in ["ap@10", "precision@1", "precision@10", "negative_precision@10"]:
            delta = np.asarray([x[metric] for x in values[method]]) - np.asarray(
                [x[metric] for x in values["dual"]]
            )
            paired[backend][comparison][metric] = {
                "difference_pp": float(delta.mean() * 100),
                "reference_cluster_95ci_pp": np.percentile(
                    [delta[index].mean() * 100 for index in bootstrap], [2.5, 97.5]
                ).tolist(),
                "query_gains": int((delta > 0).sum()),
                "query_losses": int((delta < 0).sum()),
            }

    eligible_auc = [value for value in per_query_auc if value is not None]
    auc_bootstrap = []
    mean_difference_bootstrap = []
    for indices in bootstrap:
        auc_values = [per_query_auc[index] for index in indices if per_query_auc[index] is not None]
        auc_bootstrap.append(float(np.mean(auc_values)) if auc_values else float("nan"))
        positive = [
            per_query_positive_lifts[index]
            for index in indices
            if per_query_positive_lifts[index] is not None
        ]
        negative = [
            per_query_negative_lifts[index]
            for index in indices
            if per_query_negative_lifts[index] is not None
        ]
        mean_difference_bootstrap.append(
            float(np.mean(positive) - np.mean(negative))
            if positive and negative
            else float("nan")
        )
    macro_positive = [
        value for value in per_query_positive_lifts if value is not None
    ]
    macro_negative = [
        value for value in per_query_negative_lifts if value is not None
    ]
    mechanisms[backend] = {
        "source_lift_by_label_micro": {
            label: {
                "n": len(values_for_label),
                "mean": float(np.mean(values_for_label)) if values_for_label else None,
                "median": float(np.median(values_for_label)) if values_for_label else None,
            }
            for label, values_for_label in all_lifts.items()
        },
        "macro_query_auc_positive_vs_explicit_negative": {
            "n_eligible_queries": len(eligible_auc),
            "mean": float(np.mean(eligible_auc)) if eligible_auc else None,
            "reference_cluster_95ci": np.nanpercentile(auc_bootstrap, [2.5, 97.5]).tolist(),
        },
        "macro_query_mean_lift_positive_minus_explicit_negative": {
            "n_positive_queries": len(macro_positive),
            "n_negative_queries": len(macro_negative),
            "difference": float(np.mean(macro_positive) - np.mean(macro_negative)),
            "reference_cluster_95ci": np.nanpercentile(
                mean_difference_bootstrap, [2.5, 97.5]
            ).tolist(),
        },
    }

gate_checks = {
    "qwen7b_rncr_minus_dual_ap10_gt_0": paired["qwen7b_pointwise"]["rncr_alpha1 minus dual"]["ap@10"]["difference_pp"] > 0,
    "coral_rncr_minus_dual_ap10_gt_0": paired["coral_embedding"]["rncr_alpha1 minus dual"]["ap@10"]["difference_pp"] > 0,
    "qwen7b_macro_auc_gt_0_5": mechanisms["qwen7b_pointwise"]["macro_query_auc_positive_vs_explicit_negative"]["mean"] > 0.5,
    "coral_macro_auc_gt_0_5": mechanisms["coral_embedding"]["macro_query_auc_positive_vs_explicit_negative"]["mean"] > 0.5,
}
report = {
    "complete": True,
    "evidence_status": cfg["evidence_status"],
    "n_queries": len(rows),
    "n_reference_rgb_components": len(blocks),
    "summaries": summaries,
    "paired": paired,
    "mechanisms": mechanisms,
    "ties_anywhere_in_top20_scores": {
        backend: dict(counter) for backend, counter in ties.items()
    },
    "continuation_gate_evaluation": {
        "checks": gate_checks,
        "all_pass": all(gate_checks.values()),
        "meaning": cfg["continuation_gate"]["meaning"],
    },
    "config_sha256": hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
    "source_sha256": cfg["source_sha256"],
    "limits": cfg["limits"],
}
Path("runs/P034_reference_necessity_details.json").write_text(
    json.dumps(details, indent=2) + chr(10)
)
Path("results/P034_reference_necessity.json").write_text(
    json.dumps(report, indent=2) + chr(10)
)
for backend in backends:
    print(
        backend,
        "dual_ap10",
        round(summaries[backend]["dual"]["ap@10"], 6),
        "rncr_ap10",
        round(summaries[backend]["rncr_alpha1"]["ap@10"], 6),
        "delta_pp",
        round(paired[backend]["rncr_alpha1 minus dual"]["ap@10"]["difference_pp"], 6),
        "auc",
        round(mechanisms[backend]["macro_query_auc_positive_vs_explicit_negative"]["mean"], 6),
    )
print("GATE", report["continuation_gate_evaluation"])

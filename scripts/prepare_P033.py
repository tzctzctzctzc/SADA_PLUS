"""Freeze an untouched PinPoint held-out cohort before P030/P032 results exist."""
import collections
import datetime
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path("/vepfs/vlms/1")
assert Path.cwd().resolve() == ROOT

SEED = "P033-heldout-v1"
R1_SHIFTS = [3, 11, 23, 37]
R2_SHIFTS = [1, 7, 17, 31]
digest = lambda s: hashlib.sha256(s.encode()).hexdigest()
qid = lambda x: str(x).removeprefix("query_").zfill(5)
valid_sig = lambda x: isinstance(x, str) and re.fullmatch(r"[0-9a-f]{32}", x) is not None

p030_path = Path("configs/P030_pinpoint_references.json")
parquet_path = Path("data/PinPoint/pinpoint_licensed.parquet")
retrieval_path = Path("data/PinPoint/bge_vl_mllm_s1_retrieval_results_licensed.json")
p030 = json.loads(p030_path.read_text())
df = pd.read_parquet(parquet_path)
retrieval = json.loads(retrieval_path.read_text())

assert p030["new_model_scores_on_reserved_groups_seen"] is False
assert not Path("results/P030_pinpoint_references.json").exists()
assert not Path("results/P032_reference_contrast.json").exists()

multi = df[df.query_image_signature2.map(valid_sig)]
parent = {}


def find(x):
    parent.setdefault(x, x)
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]


for _, row in multi.iterrows():
    a, b = find(row.query_image_signature), find(row.query_image_signature2)
    parent[max(a, b)] = min(a, b)

groups = collections.defaultdict(list)
for _, row in multi.iterrows():
    groups[find(row.query_image_signature)].append(row)

conflicts = {
    qid(row.query_id)
    for _, row in df.iterrows()
    if set(row.positive_candidates) & set(row.negative_candidates)
}
reserved = p030["reserved_reference_components"]
eligible = {}
ineligible = []
for component in reserved:
    clean = [row for row in groups[component] if qid(row.query_id) not in conflicts]
    if clean:
        eligible[component] = clean
    else:
        ineligible.append(component)

assert len(reserved) == 60
assert len(eligible) == 54
assert len(ineligible) == 6

records = []
for component in sorted(eligible):
    row = min(
        eligible[component],
        key=lambda x: digest(SEED + "|query|" + qid(x.query_id)),
    )
    query_id = qid(row.query_id)
    assert not set(row.positive_candidates) & set(row.negative_candidates)
    assert len(retrieval[query_id]["retrieved_items"]) == 50
    records.append(
        {
            "id": query_id,
            "source_query_id": row.query_id,
            "reference_component": component,
            "query_category": row.query_category,
            "references": [
                row.query_image_signature,
                row.query_image_signature2,
            ],
            "instruction": row.instruction,
            "positive_set": sorted(x for x in set(row.positive_candidates) if x is not None),
            "negative_set": sorted(x for x in set(row.negative_candidates) if x is not None),
            "original_candidates": retrieval[query_id]["retrieved_items"],
            "candidates_to_score": retrieval[query_id]["retrieved_items"][:20],
        }
    )

assert len(records) == len({row["reference_component"] for row in records}) == 54
order = sorted(records, key=lambda row: digest(SEED + "|donor-order|" + row["id"]))
index = {row["id"]: i for i, row in enumerate(order)}
for row in records:
    i = index[row["id"]]
    donors = []
    for k, (r1_shift, r2_shift) in enumerate(zip(R1_SHIFTS, R2_SHIFTS)):
        r1_donor = order[(i + r1_shift) % len(order)]
        r2_donor = order[(i + r2_shift) % len(order)]
        assert len({row["id"], r1_donor["id"], r2_donor["id"]}) == 3
        assert not set(row["references"]) & {
            r1_donor["references"][0],
            r2_donor["references"][1],
        }
        donors.append(
            {
                "k": k,
                "r1_donor_id": r1_donor["id"],
                "r1": r1_donor["references"][0],
                "r2_donor_id": r2_donor["id"],
                "r2": r2_donor["references"][1],
            }
        )
    row["donors"] = donors

noncomposition_pool = []
for component in sorted(eligible):
    for row in eligible[component]:
        if row.query_category != "Composition":
            noncomposition_pool.append(
                {
                    "id": qid(row.query_id),
                    "reference_component": component,
                    "query_category": row.query_category,
                }
            )
noncomposition_pool.sort(key=lambda row: (row["reference_component"], row["query_category"], row["id"]))
assert len(noncomposition_pool) == 102
assert len({row["reference_component"] for row in noncomposition_pool}) == 2

category_counts = dict(collections.Counter(row["query_category"] for row in records))
pool_category_counts = dict(collections.Counter(row["query_category"] for row in noncomposition_pool))
assert category_counts == {"Composition": 53, "Context Fit": 1}
assert pool_category_counts == {"Complement": 2, "Context Fit": 94, "Swap": 6}

required_images = sorted(
    {
        sig
        for row in records
        for sig in (
            row["references"]
            + row["candidates_to_score"]
            + [donor["r1"] for donor in row["donors"]]
            + [donor["r2"] for donor in row["donors"]]
        )
    }
)
conditions_per_query = 1 + 2 + 3 * len(R1_SHIFTS)
config = {
    "id": "P033",
    "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "status": "prospectively_frozen_metadata_only_not_authorized_for_scoring",
    "purpose": "Untouched component-held-out confirmation of P032 plus a fixed multi-sham symmetric source-contrast diagnostic.",
    "seed": SEED,
    "source_experiment": "P030",
    "source_config": str(p030_path),
    "source_config_sha256": hashlib.sha256(p030_path.read_bytes()).hexdigest(),
    "source_sha256": {
        str(parquet_path): hashlib.sha256(parquet_path.read_bytes()).hexdigest(),
        str(retrieval_path): hashlib.sha256(retrieval_path.read_bytes()).hexdigest(),
    },
    "gpu_uuid_if_authorized": p030["gpu_uuid"],
    "execution_gate": {
        "result": "results/P032_reference_contrast.json",
        "required_json_path": "confirmatory_gate_evaluation.all_pass",
        "required_value": True,
        "rule": "Do not fetch held-out images, run model scores, or alter this cohort unless the frozen P032 gate passes.",
    },
    "cohort": {
        "reserved_components_total": 60,
        "eligible_components": 54,
        "ineligible_all_conflicting_components": ineligible,
        "selection": "One clean query per untouched reserved reference component, minimizing SHA256(seed|query|query_id).",
        "n_queries": len(records),
        "category_counts": category_counts,
        "expected_conditions_per_query": conditions_per_query,
        "expected_score_records": len(records) * 20 * conditions_per_query,
    },
    "annotation_policy": "Drop null padding from annotation arrays, retain every non-null ID, and use deduplicated positive/negative sets.",
    "conditions": {
        "valid_dual": "(r1,r2)",
        "valid_single_r1": "(r1)",
        "valid_single_r2": "(r2)",
        "for_each_k": [
            "sham_r2_k=(r1,b_k)",
            "sham_r1_k=(a_k,r2)",
            "sham_both_k=(a_k,b_k)",
        ],
        "k": len(R1_SHIFTS),
        "balanced_donor_shifts": {
            "r1": R1_SHIFTS,
            "r2": R2_SHIFTS,
        },
    },
    "fixed_scores": {
        "pilot_replication": "2*m(valid_dual)-m(sham_r2_0)",
        "pilot_reference_lift": "m(valid_dual)-m(sham_r2_0)",
        "symmetric_multisham": "2*m(valid_dual)-0.5*(mean_k m(sham_r1_k)+mean_k m(sham_r2_k))",
        "factorial_interaction": "mean_k[m(valid_dual)-m(sham_r1_k)-m(sham_r2_k)+m(sham_both_k)]",
        "interaction_augmented": "m(valid_dual)+factorial_interaction",
        "m": "raw True-minus-False logit margin",
    },
    "confirmatory_hierarchy": {
        "primary": "pilot_replication minus valid_dual on canonical-set AP@10",
        "primary_pass": "reference-RGB-component bootstrap 95% interval lower endpoint strictly greater than 0",
        "mechanism_checks": [
            "pilot_reference_lift positive-vs-explicit-negative macro query AUC bootstrap 95% interval lower endpoint strictly greater than 0.5",
            "positive minus explicit-negative mean pilot_reference_lift bootstrap 95% interval lower endpoint strictly greater than 0",
        ],
        "secondary": [
            "symmetric_multisham minus valid_dual",
            "interaction_augmented minus valid_dual",
            "valid_dual minus each single-reference baseline",
        ],
        "reporting_rule": "Evaluate in this order without tuning. Secondary results cannot rescue a failed primary replication.",
    },
    "interpretation": {
        "estimand": "A pre-fixed source-specific contrast in model margin under unrelated reference substitutions.",
        "assumption": "Balanced donor permutations estimate nuisance response only if unrelated reference donors are exchangeable for that nuisance.",
        "forbidden_claims": [
            "causal effect of the true second image",
            "generalization beyond Composition",
            "full PinPoint SOTA",
        ],
    },
    "novelty_boundary": [
        "RMCD already forms relevance-weighted signed sums of logits from multiple relevant and irrelevant contexts; generic multi-context subtraction is prior art.",
        "MultiSHAP already uses Shapley interaction indices for model-agnostic multimodal synergy attribution; a generic factorial interaction score is prior art.",
        "Any surviving claim must be specific to source-bound multi-reference retrieval and supported by held-out ranking gains, explicit-negative discrimination, and cross-model evidence.",
    ],
    "noncomposition_sensitivity_pool": {
        "status": "descriptive_only_not_an_independent_confirmatory_cohort",
        "reason": "All 102 clean non-Composition rows occur in only two reference components.",
        "category_counts": pool_category_counts,
        "records": noncomposition_pool,
    },
    "required_image_signatures_if_gate_passes": required_images,
    "records": records,
}
out = Path("configs/P033_heldout_reference_contrast.json")
assert not out.exists()
out.write_text(json.dumps(config, indent=2) + chr(10))
print(
    {
        "config": str(out),
        "n_queries": len(records),
        "category_counts": category_counts,
        "noncomposition_pool": pool_category_counts,
        "required_images": len(required_images),
        "expected_score_records": config["cohort"]["expected_score_records"],
        "source_config_sha256": config["source_config_sha256"],
    }
)

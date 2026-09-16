"""Freeze positive author-annotation slices without inspecting model outcomes."""
import collections, hashlib, json
from pathlib import Path
import yaml

source = Path("data/ImageCoDe/github/data/analysis/manual_annotation_valid.yaml")
raw = source.read_bytes()
assert hashlib.sha256(raw).hexdigest() == "39cfbd1cc81d3811bcb346bf4b84521ef1e8ae3641dcb1d05b397454d702b8a4"
annotations = yaml.safe_load(raw)
cfg_path = Path("configs/P023_imagecode.json")
rows = json.loads(cfg_path.read_text())["evaluation"]
slices = collections.defaultdict(list)
records = []
for row in rows:
    item = annotations.get(row["directory"], {}).get(str(row["gold"]))
    codes = []
    if item is not None:
        assert item["description"] == row["caption"], row["id"]
        codes = sorted({t.strip() for t in item.get("annotation", "").split(",") if t.strip()})
    status = "missing" if item is None else ("positive_codes" if codes else "blank")
    records.append({"id": row["id"], "kind": row["kind"], "status": status, "codes": codes})
    for code in codes:
        slices[code].append(row["id"])
out = {
    "id": "P026", "purpose": "Author-positive phenomenon slices for P023/P025; exploratory diagnostic, no new benchmark.",
    "source_url": "https://raw.githubusercontent.com/McGill-NLP/imagecode/609f07611aed2599f946c30d730f40a41af1079b/data/analysis/manual_annotation_valid.yaml",
    "source_sha256": hashlib.sha256(raw).hexdigest(),
    "P023_config_sha256": hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
    "n_evaluation": len(rows),
    "status_counts": dict(collections.Counter(r["status"] for r in records)),
    "positive_slices": dict(sorted(slices.items())),
    "primary_diagnostic_slice": "con",
    "comparisons": [
        ["patch_attention_calibrated", "caption_logprob_mean"],
        ["patch_density_calibrated", "caption_logprob_mean"],
        ["raw_logits", "caption_logprob_mean"],
        ["patch_attention_calibrated", "matching_score"]
    ],
    "bootstrap": {"replicates": 5000, "seed": 17026, "unit": "P025 source-video/pixel-overlap connected component"},
    "limitations": [
        "Only explicit positive annotation codes define slices. Missing/blank/absent codes are not negative labels.",
        "Author context labels include gradability as well as other-frame grounding; this is not an irreducible-context gold label.",
        "Small overlapping slices are diagnostic only, with no hyperparameter tuning or selection based on their results.",
        "The frozen 256-question validation pilot is not the complete ImageCoDe evaluation."
    ],
    "records": records
}
Path("configs/P026_annotation_slices.json").write_text(json.dumps(out, indent=2))
print(json.dumps({"status_counts": out["status_counts"], "positive_slice_sizes": {k:len(v) for k,v in out["positive_slices"].items()}}, indent=2))

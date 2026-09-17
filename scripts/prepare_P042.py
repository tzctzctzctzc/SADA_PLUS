#!/usr/bin/env python3
"""Freeze P042 official-triplet candidates before inspecting image content."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path("/vepfs/vlms/1")
CONFIG_PATH = ROOT / "configs/P042_role_conditioned_equivariance.json"
QUERY_PATTERN = re.compile(
    r"the same\s+(.+?)\s+(?:with|as(?:\s+in)?)\s+<Product\s+(\d+)>",
    re.IGNORECASE | re.DOTALL,
)


def stable_key(*parts: Any) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


def parse_keys(item: dict[str, Any]) -> list[str] | None:
    sources = item.get("query", [])
    found = {
        int(position) - 1: key.strip().lower()
        for key, position in QUERY_PATTERN.findall(item.get("query instruction", ""))
    }
    if len(found) != len(sources):
        return None
    return [found.get(index, "") for index in range(len(sources))]


assert Path.cwd().resolve() == ROOT
config = json.loads(CONFIG_PATH.read_text())
assert config["status"] == "stage0_data_contract_only"
run_dir = ROOT / "runs/P042_role_conditioned_equivariance"
run_dir.mkdir(parents=True, exist_ok=True)

metadata = {
    row["idx"]: row
    for row in (
        json.loads(line)
        for line in (ROOT / config["data"]["metadata_path"]).open()
        if line.strip()
    )
}
metadata_manifest = json.loads(
    (ROOT / config["data"]["metadata_manifest"]).read_text()
)
assert metadata_manifest["dataset_revision"] == config["data"]["dataset_revision"]
assert len(metadata_manifest["selected_shards"]) == config["data"]["selected_shards"]

p041_dir = ROOT / "runs/P041_selective_dependence"
p041_bundles = {
    row["bundle_id"]: row
    for row in (
        json.loads(line)
        for line in (p041_dir / "bundles.jsonl").open()
        if line.strip()
    )
}
p041_audit = json.loads((p041_dir / "data_contract_audit.json").read_text())["rows"]
blocked = {
    product_id
    for audit_row in p041_audit
    for product_id in p041_bundles[audit_row["bundle_id"]]["product_ids"]
}
assert len(blocked) == 600

allowlist = set(config["data"]["other_role_allowlist"])
counts = Counter()
eligible = []
query_path = ROOT / config["data"]["query_path"]
assert hashlib.sha256(query_path.read_bytes()).hexdigest() == config["data"]["query_sha256"]
for line in query_path.open():
    item = json.loads(line)
    source_ids = [str(value) for value in item.get("query", [])]
    positives = [str(value) for value in item.get("pos_candidate", [])]
    if len(source_ids) != 2 or len(positives) != 1:
        continue
    keys = parse_keys(item)
    if not keys or keys.count("color") != 1 or keys[0] == keys[1]:
        continue
    other_slot = 1 - keys.index("color")
    other_key = keys[other_slot]
    if other_key not in allowlist:
        continue
    counts["semantic_pattern"] += 1
    product_ids = source_ids + positives
    if any(product_id not in metadata for product_id in product_ids):
        continue
    counts["in_fixed_metadata_shards"] += 1
    if set(product_ids) & blocked:
        continue
    counts["disjoint_from_P041_audit"] += 1
    color_slot = keys.index("color")
    eligible.append(
        {
            "bundle_id": f"merit-{item['idx']}",
            "query_idx": item["idx"],
            "instruction": item["query instruction"],
            "source_ids": source_ids,
            "positive_target_id": positives[0],
            "color_slot": color_slot,
            "other_slot": other_slot,
            "color_source_id": source_ids[color_slot],
            "other_source_id": source_ids[other_slot],
            "other_key": other_key,
            "product_ids": product_ids,
            "source_locations": {
                product_id: {
                    "shard": metadata[product_id]["shard"],
                    "row_index": metadata[product_id]["row_index"],
                }
                for product_id in product_ids
            },
            "selection_hash": stable_key("P042", item["idx"]),
        }
    )

eligible.sort(key=lambda row: row["selection_hash"])
selected = []
used: set[str] = set()
for row in eligible:
    if set(row["product_ids"]) & used:
        continue
    selected.append(row)
    used.update(row["product_ids"])
    if len(selected) == config["data"]["raw_candidate_bundles"]:
        break
if len(selected) != config["data"]["raw_candidate_bundles"]:
    raise RuntimeError(f"only {len(selected)} disjoint candidates available")

out_path = run_dir / "raw_candidates.jsonl"
with out_path.open("w") as handle:
    for row in selected:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")

selected_ids = [product_id for row in selected for product_id in row["product_ids"]]
assert len(selected_ids) == len(set(selected_ids)) == 3 * len(selected)
assert not set(selected_ids) & blocked
summary = {
    "status": "pass",
    "dataset_revision": metadata_manifest["dataset_revision"],
    "query_sha256": config["data"]["query_sha256"],
    "counts": counts,
    "eligible_before_product_disjoint_greedy": len(eligible),
    "raw_candidates": len(selected),
    "unique_products": len(selected_ids),
    "P041_blocked_products": len(blocked),
    "overlap_with_P041_audit": len(set(selected_ids) & blocked),
    "other_key_counts": Counter(row["other_key"] for row in selected),
    "raw_candidates_path": str(out_path.relative_to(ROOT)),
}
(run_dir / "candidate_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
)
print(json.dumps(summary, ensure_ascii=False, indent=2))

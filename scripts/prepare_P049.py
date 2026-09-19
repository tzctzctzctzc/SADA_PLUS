#!/usr/bin/env python3
"""Freeze P049 with new training products and exact P048 held-out rows."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path("/vepfs/vlms/1")
CONFIG_PATH = ROOT / "configs/P049_counterfactual_role_verifier.json"
RUN_DIR = ROOT / "runs/P049_counterfactual_role_verifier"
QUERY_PATTERN = re.compile(r"the same\s+(.+?)\s+(?:with|as(?:\s+in)?)\s+<Product\s+(\d+)>", re.I | re.S)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(*parts: Any) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


def canonical(value: Any) -> str:
    if isinstance(value, str):
        return value.strip().lower()
    return json.dumps(value, ensure_ascii=False, sort_keys=True).lower()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open() if line.strip()]


def parse_keys(item: dict[str, Any]) -> list[str] | None:
    found = {int(position) - 1: key.strip().lower() for key, position in QUERY_PATTERN.findall(item.get("query instruction", ""))}
    keys = [found.get(0, ""), found.get(1, "")]
    return keys if all(keys) else None


def instructions(target_class: str, keys: list[str]) -> dict[str, str]:
    return {
        "original": f"Find a product of {target_class} that has the same {keys[0]} as Product 1 and the same {keys[1]} as Product 2.",
        "paraphrase": f"Retrieve a {target_class} product matching Product 2 in {keys[1]} and Product 1 in {keys[0]}.",
        "text_role_swap": f"Find a product of {target_class} that has the same {keys[0]} as Product 2 and the same {keys[1]} as Product 1.",
        "paraphrase_text_role_swap": f"Retrieve a {target_class} product matching Product 1 in {keys[1]} and Product 2 in {keys[0]}.",
    }


def main() -> None:
    assert Path.cwd().resolve() == ROOT
    config = json.loads(CONFIG_PATH.read_text())
    data = config["data"]
    query_path = ROOT / data["query_path"]
    p048_path = ROOT / data["p048_bundles"]
    assert sha256_file(query_path) == data["query_sha256"]
    assert sha256_file(p048_path) == data["p048_bundles_sha256"]
    manifest = json.loads((ROOT / data["metadata_manifest"]).read_text())
    assert manifest["dataset_revision"] == data["dataset_revision"]
    metadata = {row["idx"]: row for row in load_jsonl(ROOT / data["metadata_path"])}
    p048 = load_jsonl(p048_path)
    heldout = [dict(row) for row in p048 if row["split"] in {"validation", "confirmation"}]
    assert Counter(row["split"] for row in heldout) == Counter({"validation": 128, "confirmation": 64})
    for row in heldout:
        row["instructions"] = instructions(row["target_class"], row["role_keys"])
    heldout_products = {product_id for row in heldout for product_id in map(str, row["product_ids"])}
    role_allowlist = set(data["role_allowlist"])
    active_allowlist = set(data["active_role_allowlist"])
    excluded_products = {
        product_id
        for row in load_jsonl(ROOT / data["excluded_confirmation_source"])
        if row["split"] == "confirmation"
        for product_id in map(str, row["product_ids"])
    }
    candidates = []
    for line in query_path.open():
        item = json.loads(line)
        source_ids = list(map(str, item.get("query", [])))
        positives = list(map(str, item.get("pos_candidate", [])))
        negatives = list(map(str, item.get("hard_negative", [])))
        if len(source_ids) != 2 or len(positives) != 1 or len(negatives) != 1:
            continue
        keys = parse_keys(item)
        if not keys or keys[0] == keys[1] or any(key not in role_allowlist for key in keys):
            continue
        product_ids = source_ids + positives + negatives
        if len(set(product_ids)) != 4 or any(product_id not in metadata for product_id in product_ids):
            continue
        if set(product_ids) & (excluded_products | heldout_products):
            continue
        sources = [metadata[product_id] for product_id in source_ids]
        target = metadata[positives[0]]
        if any(key not in source["attribute"] or key not in target["attribute"] for source, key in zip(sources, keys)):
            continue
        if any(canonical(source["attribute"][key]) != canonical(target["attribute"][key]) for source, key in zip(sources, keys)):
            continue
        mismatched = [
            role for role in (0, 1)
            if keys[role] in active_allowlist
            and keys[role] in sources[1 - role]["attribute"]
            and canonical(sources[1 - role]["attribute"][keys[role]]) != canonical(target["attribute"][keys[role]])
        ]
        if not mismatched:
            continue
        active_role = mismatched[int(stable_key(data["selection_seed"], item["idx"], "role"), 16) % len(mismatched)]
        candidates.append({
            "bundle_id": f"merit-{item['idx']}-role-swap",
            "query_idx": item["idx"],
            "target_class": target["class"],
            "instructions": instructions(target["class"], keys),
            "source_ids": source_ids,
            "positive_target_id": positives[0],
            "negative_target_id": negatives[0],
            "role_keys": keys,
            "verified_mismatched_roles": mismatched,
            "active_role": active_role,
            "mirrored_from_official_order": False,
            "active_role_key": keys[active_role],
            "active_correct_source_id": source_ids[active_role],
            "active_wrong_source_id": source_ids[1 - active_role],
            "target_role_value": target["attribute"][keys[active_role]],
            "correct_source_role_value": sources[active_role]["attribute"][keys[active_role]],
            "wrong_source_role_value": sources[1 - active_role]["attribute"][keys[active_role]],
            "product_ids": product_ids,
            "source_locations": {product_id: {"shard": metadata[product_id]["shard"], "row_index": metadata[product_id]["row_index"]} for product_id in product_ids},
            "selection_hash": stable_key(data["selection_seed"], item["idx"]),
        })
    candidates.sort(key=lambda row: row["selection_hash"])
    selected_train = []
    used_products: set[str] = set()
    class_counts = Counter()
    for row in candidates:
        if set(row["product_ids"]) & used_products:
            continue
        if class_counts[row["target_class"]] >= data["max_train_bundles_per_class"]:
            continue
        copy = dict(row)
        copy["split"] = "train"
        selected_train.append(copy)
        used_products.update(row["product_ids"])
        class_counts[row["target_class"]] += 1
        if len(selected_train) == data["counts"]["train"]:
            break
    if len(selected_train) != data["counts"]["train"]:
        raise RuntimeError(f"could only select {len(selected_train)}/{data['counts']['train']} disjoint train rows")
    for index, row in enumerate(selected_train):
        desired_role = index % 2
        if row["active_role"] != desired_role:
            row["source_ids"] = list(reversed(row["source_ids"]))
            row["role_keys"] = list(reversed(row["role_keys"]))
            row["verified_mismatched_roles"] = sorted(1 - role for role in row["verified_mismatched_roles"])
            row["active_role"] = 1 - row["active_role"]
            row["mirrored_from_official_order"] = True
            row["instructions"] = instructions(row["target_class"], row["role_keys"])
        assert row["active_role"] == desired_role
    selected = selected_train + heldout
    all_products = [product_id for row in selected for product_id in row["product_ids"]]
    assert len(all_products) == len(set(all_products))
    assert Counter(row["split"] for row in selected) == Counter(data["counts"])
    assert not ({product_id for row in selected_train for product_id in row["product_ids"]} & heldout_products)
    for row in selected:
        assert row["instructions"]["original"].count("Product 1") == row["instructions"]["original"].count("Product 2") == 1
        assert row["instructions"]["text_role_swap"].count("Product 1") == row["instructions"]["text_role_swap"].count("Product 2") == 1
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    bundle_path = RUN_DIR / "bundles.jsonl"
    with bundle_path.open("w") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    visual_summary = json.loads((ROOT / data["p048_visual_audit"]).read_text())
    assert visual_summary["status"] == "pass"
    summary = {
        "status": "pass",
        "candidate_pool_after_heldout_exclusion": len(candidates),
        "bundles": len(selected),
        "split_counts": Counter(row["split"] for row in selected),
        "class_counts": {split: Counter(row["target_class"] for row in selected if row["split"] == split) for split in data["counts"]},
        "active_role_counts": Counter(row["active_role"] for row in selected),
        "unique_products": len(set(all_products)),
        "train_heldout_product_overlap": 0,
        "within_pilot_product_overlap": 0,
        "heldout_identical_to_P048": True,
        "inherited_visual_gate": visual_summary,
        "bundles_sha256": sha256_file(bundle_path),
    }
    (RUN_DIR / "prepare_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

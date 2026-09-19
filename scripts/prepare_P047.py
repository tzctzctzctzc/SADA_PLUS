#!/usr/bin/env python3
"""Freeze a large, product-disjoint P047 role-binding pilot from MERIT metadata."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path("/vepfs/vlms/1")
CONFIG_PATH = ROOT / "configs/P047_position_controlled_role_contrast.json"
RUN_DIR = ROOT / "runs/P047_position_controlled_role_contrast"
QUERY_PATTERN = re.compile(
    r"the same\s+(.+?)\s+(?:with|as(?:\s+in)?)\s+<Product\s+(\d+)>",
    re.IGNORECASE | re.DOTALL,
)


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


def parse_keys(item: dict[str, Any]) -> list[str] | None:
    found = {
        int(position) - 1: key.strip().lower()
        for key, position in QUERY_PATTERN.findall(item.get("query instruction", ""))
    }
    keys = [found.get(0, ""), found.get(1, "")]
    return keys if all(keys) else None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open() if line.strip()]


def historical_products() -> set[str]:
    products: set[str] = set()
    for path in [
        ROOT / "runs/P041_selective_dependence/bundles.jsonl",
        ROOT / "runs/P042_role_conditioned_equivariance/raw_candidates.jsonl",
        ROOT / "runs/P043_role_conditioned_equivariance_maskaware_confirm/raw_candidates.jsonl",
        ROOT / "runs/P046_self_normalized_role_corruption/bundles.jsonl",
    ]:
        if path.exists():
            for row in load_jsonl(path):
                products.update(map(str, row["product_ids"]))
    return products


def instructions(target_class: str, keys: list[str]) -> dict[str, str]:
    return {
        "original": (
            f"Find a product of {target_class} that has the same {keys[0]} as Product 1 "
            f"and the same {keys[1]} as Product 2."
        ),
        "paraphrase": (
            f"Retrieve a {target_class} product matching Product 2 in {keys[1]} and "
            f"Product 1 in {keys[0]}."
        ),
        "text_role_swap": (
            f"Find a product of {target_class} that has the same {keys[0]} as Product 2 "
            f"and the same {keys[1]} as Product 1."
        ),
    }


def main() -> None:
    assert Path.cwd().resolve() == ROOT
    config = json.loads(CONFIG_PATH.read_text())
    query_path = ROOT / config["data"]["query_path"]
    assert sha256_file(query_path) == config["data"]["query_sha256"]
    metadata_manifest = json.loads((ROOT / config["data"]["metadata_manifest"]).read_text())
    assert metadata_manifest["dataset_revision"] == config["data"]["dataset_revision"]
    metadata = {row["idx"]: row for row in load_jsonl(ROOT / config["data"]["metadata_path"])}
    role_allowlist = set(config["data"]["role_allowlist"])
    counters = Counter()
    candidates: list[dict[str, Any]] = []

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
        sources = [metadata[product_id] for product_id in source_ids]
        target = metadata[positives[0]]
        if any(key not in source["attribute"] or key not in target["attribute"] for source, key in zip(sources, keys)):
            continue
        if any(canonical(source["attribute"][key]) != canonical(target["attribute"][key]) for source, key in zip(sources, keys)):
            continue
        counters["official_role_matches"] += 1
        mismatched_roles = [
            role
            for role in (0, 1)
            if keys[role] in sources[1 - role]["attribute"]
            and canonical(sources[1 - role]["attribute"][keys[role]])
            != canonical(target["attribute"][keys[role]])
        ]
        if not mismatched_roles:
            continue
        counters["has_verified_cross_role_mismatch"] += 1
        active_role = mismatched_roles[
            int(stable_key(config["data"]["selection_seed"], item["idx"], "role"), 16)
            % len(mismatched_roles)
        ]
        desired_role_position = int(
            stable_key(config["data"]["selection_seed"], item["idx"], "role-position"), 16
        ) % 2
        mirrored = active_role != desired_role_position
        if mirrored:
            source_ids.reverse()
            sources.reverse()
            keys.reverse()
            mismatched_roles = sorted(1 - role for role in mismatched_roles)
            active_role = 1 - active_role
        assert active_role == desired_role_position
        generated = instructions(target["class"], keys)
        requested_values = [canonical(target["attribute"][key]) for key in keys]
        if any(len(value) >= 3 and any(value in text.lower() for text in generated.values()) for value in requested_values):
            continue
        candidates.append(
            {
                "bundle_id": f"merit-{item['idx']}-role-swap",
                "query_idx": item["idx"],
                "target_class": target["class"],
                "instructions": generated,
                "source_ids": source_ids,
                "positive_target_id": positives[0],
                "negative_target_id": negatives[0],
                "role_keys": keys,
                "verified_mismatched_roles": mismatched_roles,
                "active_role": active_role,
                "mirrored_from_official_order": mirrored,
                "active_role_key": keys[active_role],
                "active_correct_source_id": source_ids[active_role],
                "active_wrong_source_id": source_ids[1 - active_role],
                "target_role_value": target["attribute"][keys[active_role]],
                "correct_source_role_value": sources[active_role]["attribute"][keys[active_role]],
                "wrong_source_role_value": sources[1 - active_role]["attribute"][keys[active_role]],
                "product_ids": product_ids,
                "source_locations": {
                    product_id: {
                        "shard": metadata[product_id]["shard"],
                        "row_index": metadata[product_id]["row_index"],
                    }
                    for product_id in product_ids
                },
                "selection_hash": stable_key(config["data"]["selection_seed"], item["idx"]),
            }
        )

    candidates.sort(key=lambda row: row["selection_hash"])
    globally_disjoint: list[dict[str, Any]] = []
    used: set[str] = set()
    for row in candidates:
        if set(row["product_ids"]) & used:
            continue
        globally_disjoint.append(row)
        used.update(row["product_ids"])

    assigned: set[str] = set()
    selected: list[dict[str, Any]] = []
    required = config["data"]["counts"]
    for split in ["confirmation", "validation", "train"]:
        wanted = required[split]
        cap = config["data"]["max_bundles_per_class_per_split"][split]
        class_counts = Counter()
        role_counts = Counter()
        rows = []
        for row in globally_disjoint:
            if (
                row["bundle_id"] in assigned
                or class_counts[row["target_class"]] >= cap
                or role_counts[row["active_role"]] >= wanted // 2
            ):
                continue
            copy = dict(row)
            copy["split"] = split
            rows.append(copy)
            assigned.add(row["bundle_id"])
            class_counts[row["target_class"]] += 1
            role_counts[row["active_role"]] += 1
            if len(rows) == wanted:
                break
        if len(rows) != wanted:
            raise RuntimeError(
                f"could only create {len(rows)}/{wanted} rows for {split}; "
                f"globally disjoint pool={len(globally_disjoint)}, class_counts={dict(class_counts)}"
            )
        selected.extend(rows)

    all_products = [product_id for row in selected for product_id in row["product_ids"]]
    assert len(all_products) == len(set(all_products))
    assert Counter(row["split"] for row in selected) == Counter(required)
    for row in selected:
        assert row["active_role"] in row["verified_mismatched_roles"]
        assert canonical(row["correct_source_role_value"]) == canonical(row["target_role_value"])
        assert canonical(row["wrong_source_role_value"]) != canonical(row["target_role_value"])
        assert row["instructions"]["original"].count("Product 1") == 1
        assert row["instructions"]["original"].count("Product 2") == 1
        assert row["instructions"]["text_role_swap"].count("Product 1") == 1
        assert row["instructions"]["text_role_swap"].count("Product 2") == 1

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    bundle_path = RUN_DIR / "bundles.jsonl"
    with bundle_path.open("w") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    historical = historical_products()
    summary = {
        "status": "pass",
        "dataset_revision": metadata_manifest["dataset_revision"],
        "query_sha256": sha256_file(query_path),
        "official_role_matches": counters["official_role_matches"],
        "cross_role_candidates": len(candidates),
        "globally_disjoint_pool": len(globally_disjoint),
        "bundles": len(selected),
        "split_counts": Counter(row["split"] for row in selected),
        "class_counts": {split: Counter(row["target_class"] for row in selected if row["split"] == split) for split in required},
        "active_role_counts": Counter(row["active_role"] for row in selected),
        "verified_roles_per_bundle": Counter(len(row["verified_mismatched_roles"]) for row in selected),
        "unique_products": len(all_products),
        "historical_overlap_products": len(set(all_products) & historical),
        "within_pilot_product_overlap": len(all_products) - len(set(all_products)),
        "bundles_sha256": sha256_file(bundle_path),
    }
    (RUN_DIR / "prepare_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

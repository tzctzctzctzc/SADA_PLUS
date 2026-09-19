#!/usr/bin/env python3
"""Freeze a product-disjoint P046 role-corruption pilot from MERIT metadata."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path("/vepfs/vlms/1")
CONFIG_PATH = ROOT / "configs/P046_self_normalized_role_corruption.json"
RUN_DIR = ROOT / "runs/P046_self_normalized_role_corruption"
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
    sources = item.get("query", [])
    found = {
        int(position) - 1: key.strip().lower()
        for key, position in QUERY_PATTERN.findall(item.get("query instruction", ""))
    }
    if len(found) != len(sources):
        return None
    keys = [found.get(index, "") for index in range(len(sources))]
    return keys if all(keys) else None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open() if line.strip()]


def blocked_products() -> set[str]:
    blocked: set[str] = set()
    sources = [
        ROOT / "runs/P041_selective_dependence/bundles.jsonl",
        ROOT / "runs/P042_role_conditioned_equivariance/raw_candidates.jsonl",
        ROOT / "runs/P043_role_conditioned_equivariance_maskaware_confirm/raw_candidates.jsonl",
    ]
    for path in sources:
        for row in load_jsonl(path):
            blocked.update(map(str, row["product_ids"]))
    return blocked


def donor_for(
    source: dict[str, Any],
    target: dict[str, Any],
    key: str,
    candidates: list[str],
    metadata: dict[str, dict[str, Any]],
    unavailable: set[str],
    seed_parts: tuple[Any, ...],
) -> str | None:
    target_value = canonical(target["attribute"][key])
    ranked = sorted(
        candidates,
        key=lambda product_id: (
            metadata[product_id].get("country") != source.get("country"),
            metadata[product_id].get("language") != source.get("language"),
            stable_key(*seed_parts, product_id),
        ),
    )
    for product_id in ranked:
        row = metadata[product_id]
        if product_id in unavailable:
            continue
        if canonical(row["attribute"].get(key)) == target_value:
            continue
        return product_id
    return None


def main() -> None:
    assert Path.cwd().resolve() == ROOT
    config = json.loads(CONFIG_PATH.read_text())
    query_path = ROOT / config["data"]["query_path"]
    assert sha256_file(query_path) == config["data"]["query_sha256"]
    metadata_manifest = json.loads((ROOT / config["data"]["metadata_manifest"]).read_text())
    assert metadata_manifest["dataset_revision"] == config["data"]["dataset_revision"]
    metadata = {
        row["idx"]: row
        for row in load_jsonl(ROOT / config["data"]["metadata_path"])
    }
    blocked = blocked_products()
    role_allowlist = set(config["data"]["role_allowlist"])

    postings: dict[tuple[str, str], list[str]] = defaultdict(list)
    for product_id, row in metadata.items():
        if product_id in blocked:
            continue
        for key in row["attribute"]:
            postings[(row["class"], key)].append(product_id)

    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    counters = Counter()
    for line in query_path.open():
        item = json.loads(line)
        source_ids = list(map(str, item.get("query", [])))
        positives = list(map(str, item.get("pos_candidate", [])))
        if len(source_ids) != 2 or len(positives) != 1:
            continue
        keys = parse_keys(item)
        if not keys or keys[0] == keys[1] or any(key not in role_allowlist for key in keys):
            continue
        ids = source_ids + positives
        if any(product_id not in metadata for product_id in ids):
            continue
        if set(ids) & blocked or len(set(ids)) != 3:
            continue
        target = metadata[positives[0]]
        sources = [metadata[value] for value in source_ids]
        if any(key not in source["attribute"] or key not in target["attribute"] for source, key in zip(sources, keys)):
            continue
        if any(canonical(source["attribute"][key]) != canonical(target["attribute"][key]) for source, key in zip(sources, keys)):
            continue
        corrupt_slot = int(stable_key(config["data"]["selection_seed"], item["idx"]), 16) % 2
        key = keys[corrupt_slot]
        if len(postings[(sources[corrupt_slot]["class"], key)]) < 2:
            continue
        counters["eligible_query"] += 1
        by_class[target["class"]].append(
            {
                "item": item,
                "source_ids": source_ids,
                "target_id": positives[0],
                "target_class": target["class"],
                "role_keys": keys,
                "corrupt_slot": corrupt_slot,
                "selection_hash": stable_key(config["data"]["selection_seed"], item["idx"]),
            }
        )

    for rows in by_class.values():
        rows.sort(key=lambda row: row["selection_hash"])

    required = config["data"]["counts"]
    split_order = ["confirmation", "validation", "train"]
    selected: list[dict[str, Any]] = []
    used = set(blocked)
    class_offsets = Counter()
    classes = sorted(by_class, key=lambda name: (-len(by_class[name]), name))

    for split in split_order:
        wanted = required[split]
        class_cap = config["data"]["max_bundles_per_class_per_split"]
        split_rows: list[dict[str, Any]] = []
        for target_class in classes:
            rows = by_class[target_class]
            cursor = class_offsets[target_class]
            made_for_class: list[dict[str, Any]] = []
            while (
                cursor < len(rows)
                and len(split_rows) + len(made_for_class) < wanted
                and len(made_for_class) < class_cap
            ):
                base = rows[cursor]
                cursor += 1
                source_ids = base["source_ids"]
                target_id = base["target_id"]
                if set(source_ids + [target_id]) & used:
                    continue
                slot = base["corrupt_slot"]
                source = metadata[source_ids[slot]]
                target = metadata[target_id]
                key = base["role_keys"][slot]
                unavailable = used | set(source_ids) | {target_id}
                donor_id = donor_for(
                    source,
                    target,
                    key,
                    postings[(source["class"], key)],
                    metadata,
                    unavailable,
                    (split, base["item"]["idx"], slot),
                )
                if donor_id is None:
                    continue
                clean_instruction = (
                    f"Find a product of {base['target_class']} that has the same "
                    f"{base['role_keys'][0]} as Product 1 and the same "
                    f"{base['role_keys'][1]} as Product 2."
                )
                values = [
                    canonical(metadata[source_ids[index]]["attribute"][role_key])
                    for index, role_key in enumerate(base["role_keys"])
                ]
                if any(len(value) >= 3 and value in clean_instruction.lower() for value in values):
                    continue
                product_ids = source_ids + [target_id, donor_id]
                row = {
                    "bundle_id": f"merit-{base['item']['idx']}-slot{slot}",
                    "query_idx": base["item"]["idx"],
                    "split": split,
                    "target_class": base["target_class"],
                    "instruction": clean_instruction,
                    "source_ids": source_ids,
                    "positive_target_id": target_id,
                    "role_keys": base["role_keys"],
                    "corrupt_slot": slot,
                    "corrupt_role_key": key,
                    "donor_id": donor_id,
                    "source_role_value": metadata[source_ids[slot]]["attribute"][key],
                    "target_role_value": target["attribute"][key],
                    "donor_role_value": metadata[donor_id]["attribute"][key],
                    "product_ids": product_ids,
                    "source_locations": {
                        product_id: {
                            "shard": metadata[product_id]["shard"],
                            "row_index": metadata[product_id]["row_index"],
                        }
                        for product_id in product_ids
                    },
                    "selection_hash": base["selection_hash"],
                }
                made_for_class.append(row)
                used.update(product_ids)
            class_offsets[target_class] = cursor
            if len(made_for_class) == 1:
                lone = made_for_class.pop()
                used.difference_update(lone["product_ids"])
            split_rows.extend(made_for_class)
            if len(split_rows) >= wanted:
                break
        if len(split_rows) != wanted:
            raise RuntimeError(f"could only create {len(split_rows)}/{wanted} rows for {split}")

        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in split_rows:
            grouped[row["target_class"]].append(row)
        for target_class, rows in grouped.items():
            assert len(rows) >= 2
            for index, row in enumerate(rows):
                candidates = rows[index + 1 :] + rows[: index + 1]
                negative = next(
                    other for other in candidates
                    if any(
                        canonical(metadata[other["positive_target_id"]]["attribute"].get(key))
                        != canonical(metadata[row["positive_target_id"]]["attribute"].get(key))
                        for key in row["role_keys"]
                    )
                )
                row["negative_target_id"] = negative["positive_target_id"]
        selected.extend(split_rows)

    all_products = [product_id for row in selected for product_id in row["product_ids"]]
    assert len(all_products) == len(set(all_products))
    assert not set(all_products) & blocked
    assert Counter(row["split"] for row in selected) == Counter(required)
    for row in selected:
        slot = row["corrupt_slot"]
        key = row["corrupt_role_key"]
        assert row["role_keys"][slot] == key
        assert canonical(row["source_role_value"]) == canonical(row["target_role_value"])
        assert canonical(row["donor_role_value"]) != canonical(row["target_role_value"])
        assert row["negative_target_id"] != row["positive_target_id"]

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    bundle_path = RUN_DIR / "bundles.jsonl"
    with bundle_path.open("w") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "status": "pass",
        "dataset_revision": metadata_manifest["dataset_revision"],
        "query_sha256": sha256_file(query_path),
        "blocked_products": len(blocked),
        "eligible_queries": counters["eligible_query"],
        "bundles": len(selected),
        "split_counts": Counter(row["split"] for row in selected),
        "class_counts": Counter(row["target_class"] for row in selected),
        "role_counts": Counter(row["corrupt_role_key"] for row in selected),
        "unique_base_products": len(all_products),
        "overlap_with_prior": len(set(all_products) & blocked),
        "bundles_sha256": sha256_file(bundle_path),
    }
    (RUN_DIR / "prepare_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prepare and audit P041 intervention bundles without loading model scores.

The script reads only metadata columns from a fixed MERIT shard subset.  A bundle
uses the same A -> A' source intervention twice:
  * active query: asks for changed attribute k, so target C must change to C'
  * null query: asks for shared attribute m, so target D stays valid
This is the minimum data contract needed by the proposed mechanism.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

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


def stable_int(*parts: Any) -> int:
    text = "|".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def canonical_value(value: Any) -> str:
    if isinstance(value, str):
        return value.strip().lower()
    return json.dumps(value, sort_keys=True, ensure_ascii=False).lower()


def shard_indices(count: int, total: int) -> list[int]:
    if count < 2 or count > total:
        raise ValueError(f"invalid shard count: {count}/{total}")
    indices = [round(i * (total - 1) / (count - 1)) for i in range(count)]
    if len(set(indices)) != count:
        raise ValueError("linspace shard selection produced duplicates")
    return indices


def parse_attributes(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        attrs = raw
    else:
        try:
            attrs = json.loads(raw or "{}")
        except (TypeError, json.JSONDecodeError):
            return {}
    return {str(key).strip().lower(): value for key, value in attrs.items() if value not in (None, "", [])}


def read_one_shard(repo: str, revision: str, shard: int, total: int, endpoint: str) -> list[dict[str, Any]]:
    fs = HfFileSystem(endpoint=endpoint)
    remote = f"datasets/{repo}@{revision}/data/train-{shard:05d}-of-{total:05d}.parquet"
    with fs.open(remote, "rb") as handle:
        table = pq.read_table(
            handle,
            columns=["idx", "class", "country", "language", "attribute", "title"],
        )
    rows = table.to_pylist()
    output = []
    for row_index, row in enumerate(rows):
        output.append(
            {
                "idx": str(row["idx"]),
                "class": str(row["class"]).strip().lower(),
                "country": row["country"],
                "language": row["language"],
                "attribute": parse_attributes(row["attribute"]),
                "title": row["title"],
                "shard": shard,
                "row_index": row_index,
            }
        )
    return output


def load_metadata(config: dict[str, Any], run_dir: Path, endpoint: str, rebuild: bool) -> tuple[dict[str, dict], list[int], str]:
    metadata_path = run_dir / "products_metadata.jsonl"
    manifest_path = run_dir / "metadata_manifest.json"
    if metadata_path.exists() and manifest_path.exists() and not rebuild:
        manifest = json.loads(manifest_path.read_text())
        products = {
            row["idx"]: row
            for row in (json.loads(line) for line in metadata_path.open() if line.strip())
        }
        return products, manifest["selected_shards"], manifest["dataset_revision"]

    shard_cfg = config["data"]["selected_train_shards"]
    selected = shard_indices(shard_cfg["count"], shard_cfg["total"])
    revision = HfApi(endpoint=endpoint).dataset_info(config["data"]["dataset"]).sha
    workers = config["data"]["metadata_workers"]
    by_shard: dict[int, list[dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                read_one_shard,
                config["data"]["dataset"],
                revision,
                shard,
                shard_cfg["total"],
                endpoint,
            ): shard
            for shard in selected
        }
        for future in as_completed(futures):
            shard = futures[future]
            rows = future.result()
            by_shard[shard] = rows
            print(json.dumps({"event": "metadata_shard", "shard": shard, "rows": len(rows)}), flush=True)

    products: dict[str, dict] = {}
    with metadata_path.open("w") as handle:
        for shard in selected:
            for row in by_shard[shard]:
                products[row["idx"]] = row
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "dataset": config["data"]["dataset"],
        "dataset_revision": revision,
        "selected_shards": selected,
        "product_count": len(products),
        "metadata_path": str(metadata_path),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return products, selected, revision


def parse_query_constraints(item: dict[str, Any]) -> list[str] | None:
    source_ids = [str(value) for value in item.get("query", [])]
    matches = QUERY_PATTERN.findall(item.get("query instruction", ""))
    by_position: dict[int, str] = {}
    for key, position in matches:
        by_position[int(position) - 1] = key.strip().lower()
    if len(by_position) != len(source_ids):
        return None
    return [by_position.get(index, "") for index in range(len(source_ids))]


def intersection_candidates(
    postings: dict[tuple[str, str, str], set[str]],
    target_class: str,
    constraints: dict[str, Any],
) -> set[str]:
    sets = []
    for key, value in constraints.items():
        sets.append(postings.get((target_class, key, canonical_value(value)), set()))
    if not sets:
        return set()
    return set.intersection(*sets)


def ordered_ids(ids: Iterable[str], *seed_parts: Any) -> list[str]:
    return sorted(ids, key=lambda value: stable_int(*seed_parts, value))


def query_split(query_idx: Any) -> str:
    return "validation" if stable_int("split", query_idx) % 5 == 0 else "train"


def build_bundle(
    item: dict[str, Any],
    products: dict[str, dict],
    postings: dict[tuple[str, str, str], set[str]],
    excluded_keys: set[str],
    allowed_keys: set[str],
    forbidden_ids: set[str],
) -> dict[str, Any] | None:
    source_ids = [str(value) for value in item["query"]]
    positive_ids = [str(value) for value in item["pos_candidate"]]
    if len(source_ids) != 2 or len(positive_ids) != 1:
        return None
    if any(value not in products for value in source_ids + positive_ids):
        return None

    keys = parse_query_constraints(item)
    if not keys or any(not key or key in excluded_keys or key not in allowed_keys for key in keys):
        return None
    if keys[0] == keys[1]:
        return None

    sources = [products[value] for value in source_ids]
    positive = products[positive_ids[0]]
    for source, key in zip(sources, keys):
        if key not in source["attribute"] or key not in positive["attribute"]:
            return None
        if canonical_value(source["attribute"][key]) != canonical_value(positive["attribute"][key]):
            return None

    slot_order = [stable_int("slot", item["idx"]) % 2, 1 - stable_int("slot", item["idx"]) % 2]
    for active_slot in slot_order:
        other_slot = 1 - active_slot
        source_a = sources[active_slot]
        source_b = sources[other_slot]
        key_active = keys[active_slot]
        key_other = keys[other_slot]
        value_old = source_a["attribute"][key_active]
        value_other = source_b["attribute"][key_other]
        common_keys = [
            key
            for key in source_a["attribute"]
            if key in allowed_keys and key not in excluded_keys and key not in {key_active, key_other}
        ]
        common_keys.sort(key=lambda key: stable_int("null_key", item["idx"], active_slot, key))

        for null_key in common_keys:
            null_value = source_a["attribute"][null_key]
            alternatives = postings.get(
                (source_a["class"], null_key, canonical_value(null_value)),
                set(),
            )
            for alt_id in ordered_ids(alternatives, "alt", item["idx"], active_slot, null_key)[:200]:
                if alt_id in set(source_ids + positive_ids) | forbidden_ids:
                    continue
                alternative = products[alt_id]
                if key_active not in alternative["attribute"]:
                    continue
                value_new = alternative["attribute"][key_active]
                if canonical_value(value_new) == canonical_value(value_old):
                    continue

                active_new = intersection_candidates(
                    postings,
                    positive["class"],
                    {key_active: value_new, key_other: value_other},
                )
                null_targets = intersection_candidates(
                    postings,
                    positive["class"],
                    {null_key: null_value, key_other: value_other},
                )
                banned = set(source_ids + positive_ids + [alt_id]) | forbidden_ids
                c_prime = next(
                    (value for value in ordered_ids(active_new, "c_prime", item["idx"], alt_id) if value not in banned),
                    None,
                )
                if c_prime is None:
                    continue
                banned.add(c_prime)
                null_target = next(
                    (value for value in ordered_ids(null_targets, "null_target", item["idx"], alt_id) if value not in banned),
                    None,
                )
                if null_target is None:
                    continue

                all_ids = source_ids + [alt_id, positive_ids[0], c_prime, null_target]
                if len(set(all_ids)) != 6:
                    continue
                target_class = positive["class"]
                query_active = (
                    f"Find a {target_class} with the same {key_active} as Product 1 "
                    f"and the same {key_other} as Product 2."
                )
                query_null = (
                    f"Find a {target_class} with the same {null_key} as Product 1 "
                    f"and the same {key_other} as Product 2."
                )
                if active_slot == 1:
                    # Product 1 is always the intervened source in generated examples.
                    fixed_source_id = source_ids[0]
                else:
                    fixed_source_id = source_ids[1]
                return {
                    "bundle_id": f"merit-{item['idx']}-slot{active_slot}",
                    "source_query_idx": item["idx"],
                    "target_class": target_class,
                    "intervened_source": {
                        "before": source_ids[active_slot],
                        "after": alt_id,
                        "active_key": key_active,
                        "active_value_before": value_old,
                        "active_value_after": value_new,
                        "null_key": null_key,
                        "null_value": null_value,
                    },
                    "fixed_source": {
                        "idx": fixed_source_id,
                        "key": key_other,
                        "value": value_other,
                    },
                    "active": {
                        "instruction": query_active,
                        "target_before": positive_ids[0],
                        "target_after": c_prime,
                    },
                    "null": {
                        "instruction": query_null,
                        "target": null_target,
                    },
                    "product_ids": all_ids,
                    "source_shards": sorted({products[value]["shard"] for value in all_ids}),
                }
    return None


def validate_bundle(bundle: dict[str, Any], products: dict[str, dict]) -> list[str]:
    errors = []
    ids = bundle["product_ids"]
    if len(ids) != 6 or len(set(ids)) != 6:
        errors.append("product_ids_not_six_unique")
        return errors
    if any(value not in products for value in ids):
        errors.append("missing_product")
        return errors

    src = bundle["intervened_source"]
    before = products[src["before"]]
    after = products[src["after"]]
    fixed = products[bundle["fixed_source"]["idx"]]
    c0 = products[bundle["active"]["target_before"]]
    c1 = products[bundle["active"]["target_after"]]
    d = products[bundle["null"]["target"]]
    ka, kn, kb = src["active_key"], src["null_key"], bundle["fixed_source"]["key"]
    if before["class"] != after["class"]:
        errors.append("source_class_changed")
    if canonical_value(before["attribute"].get(ka)) == canonical_value(after["attribute"].get(ka)):
        errors.append("active_value_did_not_change")
    if canonical_value(before["attribute"].get(kn)) != canonical_value(after["attribute"].get(kn)):
        errors.append("null_value_changed")
    if canonical_value(c0["attribute"].get(ka)) != canonical_value(before["attribute"].get(ka)):
        errors.append("old_target_active_mismatch")
    if canonical_value(c1["attribute"].get(ka)) != canonical_value(after["attribute"].get(ka)):
        errors.append("new_target_active_mismatch")
    for target, label in [(c0, "old"), (c1, "new"), (d, "null")]:
        if canonical_value(target["attribute"].get(kb)) != canonical_value(fixed["attribute"].get(kb)):
            errors.append(f"{label}_target_fixed_mismatch")
    if canonical_value(d["attribute"].get(kn)) != canonical_value(before["attribute"].get(kn)):
        errors.append("null_target_mismatch")

    leaked_values = [
        canonical_value(src["active_value_before"]),
        canonical_value(src["active_value_after"]),
        canonical_value(src["null_value"]),
        canonical_value(bundle["fixed_source"]["value"]),
    ]
    text = (bundle["active"]["instruction"] + " " + bundle["null"]["instruction"]).lower()
    for value in leaked_values:
        if len(value) >= 3 and value in text:
            errors.append("attribute_value_leak")
            break
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/P041_selective_dependence_pilot.json")
    parser.add_argument("--stage", choices=["metadata", "bundles", "all"], default="all")
    parser.add_argument("--rebuild-metadata", action="store_true")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    run_dir = Path("runs/P041_selective_dependence")
    run_dir.mkdir(parents=True, exist_ok=True)
    query_path = Path(config["data"]["train_query_path"])
    actual_sha = sha256_file(query_path)
    if actual_sha != config["data"]["train_query_sha256"]:
        raise RuntimeError(f"query SHA mismatch: {actual_sha}")

    endpoint = "https://hf-mirror.com"
    products, selected_shards, dataset_revision = load_metadata(
        config, run_dir, endpoint, args.rebuild_metadata
    )
    if args.stage == "metadata":
        print(json.dumps({"status": "metadata_ready", "products": len(products)}))
        return

    postings: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for product_id, product in products.items():
        for key, value in product["attribute"].items():
            postings[(product["class"], key, canonical_value(value))].add(product_id)

    compact_queries = []
    with query_path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            ids = [str(value) for value in item.get("query", []) + item.get("pos_candidate", [])]
            if len(item.get("query", [])) == 2 and len(item.get("pos_candidate", [])) == 1:
                if all(value in products for value in ids):
                    compact_queries.append(item)

    excluded = set(config["data"]["exclude_attribute_keys"])
    allowed = set(config["data"]["visual_attribute_allowlist"])
    split_targets = {
        "validation": config["data"]["validation_bundles"],
        "train": config["data"]["train_bundles"],
    }
    bundles: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    used_ids: set[str] = set()
    rejection_counts = Counter()

    for split in ["validation", "train"]:
        candidates = [item for item in compact_queries if query_split(item["idx"]) == split]
        candidates.sort(key=lambda item: stable_int("query_order", split, item["idx"]))
        for item in candidates:
            if len(bundles[split]) >= split_targets[split]:
                break
            base_ids = {str(value) for value in item["query"] + item["pos_candidate"]}
            if base_ids & used_ids:
                rejection_counts[f"{split}:base_overlap"] += 1
                continue
            bundle = build_bundle(item, products, postings, excluded, allowed, used_ids)
            if bundle is None:
                rejection_counts[f"{split}:no_valid_bundle"] += 1
                continue
            errors = validate_bundle(bundle, products)
            if errors:
                rejection_counts[f"{split}:contract_error"] += 1
                continue
            bundle["split"] = split
            bundles[split].append(bundle)
            used_ids.update(bundle["product_ids"])

    all_bundles = bundles["train"] + bundles["validation"]
    bundle_path = run_dir / "bundles.jsonl"
    with bundle_path.open("w") as handle:
        for bundle in all_bundles:
            handle.write(json.dumps(bundle, ensure_ascii=False) + "\n")

    audit_n = config["data"]["data_audit_bundles"]
    audit_pool = sorted(all_bundles, key=lambda row: stable_int("audit", row["bundle_id"]))[:audit_n]
    audit_rows = []
    for bundle in audit_pool:
        errors = validate_bundle(bundle, products)
        audit_rows.append({"bundle_id": bundle["bundle_id"], "passed": not errors, "errors": errors})
    audit_passed = sum(row["passed"] for row in audit_rows)
    audit_rate = audit_passed / len(audit_rows) if audit_rows else 0.0

    summary = {
        "status": "pass" if (
            len(bundles["train"]) == split_targets["train"]
            and len(bundles["validation"]) == split_targets["validation"]
            and len(audit_rows) == audit_n
            and audit_rate >= 0.95
        ) else "fail",
        "dataset_revision": dataset_revision,
        "selected_shards": selected_shards,
        "metadata_products": len(products),
        "eligible_official_queries_in_subset": len(compact_queries),
        "train_bundles": len(bundles["train"]),
        "validation_bundles": len(bundles["validation"]),
        "exact_product_overlap_between_splits": len(
            set(value for row in bundles["train"] for value in row["product_ids"])
            & set(value for row in bundles["validation"] for value in row["product_ids"])
        ),
        "automated_contract_audit": {
            "sample_size": len(audit_rows),
            "passed": audit_passed,
            "pass_rate": audit_rate,
            "visual_semantic_audit": "pending_image_extraction",
        },
        "active_key_counts": Counter(
            row["intervened_source"]["active_key"] for row in all_bundles
        ).most_common(),
        "null_key_counts": Counter(
            row["intervened_source"]["null_key"] for row in all_bundles
        ).most_common(),
        "rejections": dict(rejection_counts),
        "bundle_path": str(bundle_path),
    }
    (run_dir / "data_contract_audit.json").write_text(
        json.dumps({"summary": summary, "rows": audit_rows}, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()


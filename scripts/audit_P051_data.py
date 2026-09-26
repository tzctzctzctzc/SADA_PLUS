#!/usr/bin/env python3
"""审计完整 MERIT 能否支持已确认的 pilot；不足时不缩小门槛。"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
from collections import Counter
from pathlib import Path

PATTERN = re.compile(r"the same\s+(.+?)\s+(?:with|as(?:\s+in)?)\s+<Product\s+(\d+)>", re.I)


def canonical(value):
    return str(value).strip().lower() if isinstance(value, str) else json.dumps(value, sort_keys=True).lower()


def read_jsonl(path):
    return [json.loads(line) for line in path.open() if line.strip()]


def semantic_energy(sources, target, keys, assignment):
    attrs = target["attribute"]
    if any(key not in attrs or key not in sources[source]["attribute"] for source, key in zip(assignment, keys)):
        return None
    return sum(canonical(sources[source]["attribute"][key]) == canonical(attrs[key])
               for source, key in zip(assignment, keys)) / len(keys)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/P051_tracebind.json")
    parser.add_argument("--metadata", help="可审计缓存子集，但绝不作为完整数据门槛通过")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    data = config["data"]
    meta_path = Path(args.metadata or data["metadata_path"])
    meta = {row["idx"]: row for row in read_jsonl(meta_path)}
    families = {key: family for family, keys in data["visual_families"].items() for key in keys}
    out = Path("runs/P051_tracebind")
    out.mkdir(parents=True, exist_ok=True)
    excluded = set()
    for old_path in [Path("runs/P047_position_controlled_role_contrast/bundles.jsonl"),
                     Path("runs/P048_visible_role_contrast/bundles.jsonl")]:
        if old_path.exists():
            excluded.update(product for row in read_jsonl(old_path) if row["split"] != "train"
                            for product in map(str, row["product_ids"]))
    counts, active, role_counts, source_counts = Counter(), Counter(), Counter(), Counter()
    candidates = []
    query_digest = hashlib.sha256()
    with Path(data["query_path"]).open("rb") as handle:
        for line in handle:
            query_digest.update(line)
            item = json.loads(line)
            ids = list(map(str, item.get("query", [])))
            positives, negatives = list(map(str, item.get("pos_candidate", []))), list(map(str, item.get("hard_negative", [])))
            if len(ids) not in data["source_counts"] or len(positives) != 1 or len(negatives) != 1:
                continue
            counts["official_multi_reference_single_pair"] += 1
            products = ids + positives + negatives
            if len(set(products)) != len(products) or any(product not in meta for product in products):
                continue
            counts["available_metadata"] += 1
            matches = PATTERN.findall(item.get("query instruction", ""))
            # 每来源恰好一个角色；额外条件或重复 Product 不静默丢弃。
            by_source = {int(index) - 1: key.strip().lower() for key, index in matches}
            if len(matches) != len(ids) or set(by_source) != set(range(len(ids))):
                continue
            keys = [by_source[i] for i in range(len(ids))]
            if len(set(keys)) != len(keys):
                continue
            counts["unambiguous_one_role_per_source"] += 1
            if any(key not in families for key in keys):
                continue
            counts["all_roles_visual_allowlisted"] += 1
            sources, pos, neg = [meta[product] for product in ids], meta[positives[0]], meta[negatives[0]]
            permutations = list(itertools.permutations(range(len(ids))))
            energies = [semantic_energy(sources, pos, keys, permutation) for permutation in permutations]
            if energies[0] != 1.0:
                continue
            counts["positive_metadata_consistent"] += 1
            wrong = [value is not None and value < 1.0 for value in energies]
            if not any(wrong):
                continue
            counts["some_verified_wrong_binding"] += 1
            if set(products) & excluded:
                continue
            counts["historical_heldout_products_excluded"] += 1
            neg_energies = [semantic_energy(sources, neg, keys, permutation) for permutation in permutations]
            known_wrong = [i for i, flag in enumerate(wrong) if flag]
            # DoD 的偏序必须有元数据依据；普通 hard negative 并不天然满足。
            positive_gap = 1.0 - max(energies[i] for i in known_wrong)
            negative_gap = None if neg_energies[0] is None or any(neg_energies[i] is None for i in known_wrong) else neg_energies[0] - max(neg_energies[i] for i in known_wrong)
            dod_eligible = negative_gap is not None and positive_gap > negative_gap
            if dod_eligible:
                counts["metadata_supported_dod_pair"] += 1
            mismatched_roles = sorted({role for permutation, energy in zip(permutations, energies)
                                       if energy is not None and energy < 1.0
                                       for role, source in enumerate(permutation)
                                       if canonical(sources[source]["attribute"][keys[role]]) != canonical(pos["attribute"][keys[role]])})
            # 优先非颜色角色只用于审计潜在供应量；不是看模型结果选样本。
            chosen = min(mismatched_roles, key=lambda role: (families[keys[role]] == "color", keys[role]))
            active[families[keys[chosen]]] += 1
            role_counts[keys[chosen]] += 1
            source_counts[len(ids)] += 1
            candidates.append({"bundle_id": f"merit-{item['idx']}-tracebind", "query_idx": item["idx"],
                               "source_ids": ids, "positive_target_id": positives[0], "negative_target_id": negatives[0],
                               "product_ids": products, "role_keys": keys, "active_role_key": keys[chosen],
                               "active_family": families[keys[chosen]], "target_class": pos["class"],
                               "valid_assignments": [value == 1.0 for value in energies],
                               "supervised_assignments": [value is not None for value in energies],
                               "verified_wrong_assignments": wrong, "dod_eligible": dod_eligible,
                               "semantic_assignment_energies": energies, "negative_semantic_assignment_energies": neg_energies,
                               "instruction": item["query instruction"],
                               "source_locations": {product: {"shard": meta[product]["shard"], "row_index": meta[product]["row_index"]} for product in products}})
    assert query_digest.hexdigest() == data["query_sha256"]
    # 依赖组件用于报告独立性及判断能否直接按组件划分。
    parent = {}
    def find(value):
        parent.setdefault(value, value)
        while value != parent[value]:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value
    for row in candidates:
        anchor = find(row["product_ids"][0])
        for product in row["product_ids"][1:]:
            parent[find(product)] = anchor
    components = Counter(find(row["product_ids"][0]) for row in candidates)
    ordered = sorted(candidates, key=lambda row: hashlib.sha256(str(row["query_idx"]).encode()).digest())
    validation_products, validation_count, independent_active = set(), 0, Counter()
    for row in ordered:
        if set(row["product_ids"]) & validation_products:
            continue
        validation_products.update(row["product_ids"])
        validation_count += 1
        independent_active[row["active_family"]] += 1
    n = len(candidates)
    # 必要条件上界：在角色 cap 下最多能组成多少平衡组，不能当作可行划分。
    balanced_upper = min([n] + [int((n - count) / (1.0 - data["maximum_active_role_fraction"]))
                              for count in role_counts.values()])
    supply_checks = {"raw_groups_for_requested_splits": n >= data["minimum_train_groups"] + data["minimum_validation_groups"],
                     "four_visual_families_available": len(active) >= data["minimum_visual_families"],
                     "balanced_pool_upper_bound_large_enough": balanced_upper >= data["minimum_train_groups"] + data["minimum_validation_groups"],
                     "independent_validation_upper_supply": validation_count >= data["minimum_validation_groups"]}
    manifest_path = out / ("candidate_pool_partial.jsonl" if args.metadata else "candidate_pool.jsonl")
    with manifest_path.open("w") as handle:
        for row in candidates:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {"status": "screening_only_no_visual_or_split_gate", "metadata_products": len(meta),
              "complete_metadata": not bool(args.metadata), "filter_counts": counts, "candidate_groups": n,
              "active_family_counts": active, "active_role_counts": role_counts, "source_count_distribution": source_counts,
              "unique_products": len(parent), "connected_components": len(components),
              "largest_component_groups": max(components.values(), default=0),
              "greedy_globally_product_disjoint_groups": validation_count,
              "greedy_independent_active_families": independent_active, "balanced_pool_upper_bound": balanced_upper,
              "necessary_supply_checks": supply_checks, "pilot_authorized": False,
              "remaining_gates": ["frozen feasible split", "visual intervention verification", "cross-domain correspondence join", "complete implementation runtime audit"],
              "candidate_pool_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
    (out / ("data_audit_partial.json" if args.metadata else "data_audit.json")).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

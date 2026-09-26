#!/usr/bin/env python3
"""从真实 MERIT 商品补建反事实组；属性筛选不冒充视觉核验。

每个 split 的商品池先冻结。负候选只违反 active 角色，来源换位
必须形成可验证的 DoD 偏序。三参考保留全部六种绑定，不复制排列计数。
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def canon(value):
    return value.strip().lower() if isinstance(value, str) else json.dumps(value, sort_keys=True).lower()


def stable(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big")


def instructions(cls, keys):
    return {
        "original": f"Find a product of {cls} that " + " and ".join(f"has the same {key} as Product {i+1}" for i, key in enumerate(keys)) + ".",
        "paraphrase": f"Retrieve a {cls} product matching " + ", and ".join(f"Product {i+1} in {keys[i]}" for i in reversed(range(len(keys)))) + "."
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/P051_tracebind.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    data, construction = config["data"], config["data"]["construction"]
    out = Path("runs/P051_tracebind")
    manifest = json.loads((out / "metadata_manifest.json").read_text())
    assert manifest["complete"] and manifest["dataset_revision"] == data["dataset_revision"]
    assert len(manifest["selected_shards"]) == data["total_shards"]
    assert sha(data["metadata_path"]) == manifest["sha256"]
    meta = {str(x["idx"]): x for x in map(json.loads, Path(data["metadata_path"]).open())}
    families = {key: family for family, keys in data["visual_families"].items() for key in keys}
    excluded = set()
    for path in [Path("runs/P047_position_controlled_role_contrast/bundles.jsonl"), Path("runs/P048_visible_role_contrast/bundles.jsonl")]:
        if path.exists():
            excluded.update(str(p) for row in map(json.loads, path.open()) if row["split"] != "train" for p in row["product_ids"])
    # 现有官方查询另留作迁移诊断，不进入新构造的数据。
    excluded.update(p for row in map(json.loads, (out / "candidate_pool.jsonl").open()) for p in row["product_ids"])
    pools = defaultdict(list)
    for product, row in sorted(meta.items()):
        if product in excluded:
            continue
        value = stable(construction["seed"], "product-split", product) % 100
        percentages = construction["product_split_percent"]
        assert sum(percentages.values()) == 100
        split = "train" if value < percentages["train"] else "validation" if value < percentages["train"] + percentages["validation"] else "confirmation"
        pools[split].append(row)
    all_rows, split_reports = [], {}
    for split, products in pools.items():
        rng = random.Random(stable(construction["seed"], split))
        rng.shuffle(products)
        quota = construction["candidate_counts"][split]
        per_family = quota // len(data["visual_families"])
        quota3 = round(quota * construction["three_reference_fraction"])
        usage, targets, active_counts, role_counts, n_counts = Counter(), set(), Counter(), Counter(), Counter()
        index = defaultdict(list)
        source_index = defaultdict(list)
        group_index = defaultdict(list)
        for row in products:
            attrs = row["attribute"]
            keys = sorted(key for key in attrs if key in families)
            for count in (2, 3):
                for combo in itertools.combinations(keys, count):
                    if len({families[key] for key in combo}) != count:
                        continue
                    sig = (row["class"], combo)
                    group_index[sig].append(row)
                    for active in combo:
                        source_index[(sig, active, canon(attrs[active]))].append(row)
                        rest = tuple((key, canon(attrs[key])) for key in combo if key != active)
                        index[(sig, active, rest)].append(row)
        for rows in index.values():
            rng.shuffle(rows)
        chosen = []
        max_usage = construction["maximum_product_uses"][split]
        for sweep in range(2):
            for target in products:
                if target["idx"] in targets or len(chosen) == quota:
                    continue
                attrs = target["attribute"]
                keys = sorted(key for key in attrs if key in families)
                want3 = n_counts[3] < quota3
                combos = list(itertools.combinations(keys, 3 if want3 and sweep == 0 else 2))
                rng.shuffle(combos)
                success = None
                for combo in combos:
                    if len({families[key] for key in combo}) != len(combo):
                        continue
                    active_order = sorted(combo, key=lambda key: (active_counts[families[key]], stable(target["idx"], key)))
                    sig = (target["class"], combo)
                    rows = group_index[sig]
                    for active_key in active_order:
                        family = families[active_key]
                        if active_counts[family] >= per_family:
                            continue
                        rest = tuple((key, canon(attrs[key])) for key in combo if key != active_key)
                        negative_pool = index[(sig, active_key, rest)]
                        negatives = [r for r in negative_pool if canon(r["attribute"][active_key]) != canon(attrs[active_key]) and usage[r["idx"]] < max_usage]
                        for negative in negatives[:construction["search_width"]]:
                            role_keys = list(combo)
                            rng.shuffle(role_keys)
                            source_rows = []
                            used = {target["idx"], negative["idx"]}
                            for key in role_keys:
                                eligible = [r for r in source_index[(sig, key, canon(attrs[key]))] if r["idx"] not in used and usage[r["idx"]] < max_usage
                                            and any(canon(r["attribute"][other]) != canon(attrs[other]) for other in role_keys if other != key)]
                                # 非 active 来源尽量携带负候选的 active 值，从而形成单角色换位偏序。
                                rng.shuffle(eligible)
                                eligible.sort(key=lambda r: (canon(r["attribute"][active_key]) != canon(negative["attribute"][active_key]), usage[r["idx"]]))
                                if not eligible:
                                    break
                                source_rows.append(eligible[0]); used.add(eligible[0]["idx"])
                            if len(source_rows) != len(role_keys) or usage[target["idx"]] >= max_usage:
                                continue
                            permutations = list(itertools.permutations(range(len(role_keys))))
                            def energy(candidate, assignment):
                                return sum(canon(source_rows[i]["attribute"][key]) == canon(candidate["attribute"][key]) for i, key in zip(assignment, role_keys)) / len(role_keys)
                            pos_e = [energy(target, p) for p in permutations]
                            neg_e = [energy(negative, p) for p in permutations]
                            valid = [e == 1.0 for e in pos_e]
                            active_index = role_keys.index(active_key)
                            dod_pairs = [i for i, p in enumerate(permutations) if not valid[i]
                                         and p[active_index] != active_index
                                         and canon(source_rows[p[active_index]]["attribute"][active_key]) != canon(attrs[active_key])
                                         and pos_e[0] - pos_e[i] > neg_e[0] - neg_e[i]]
                            if not dod_pairs:
                                continue
                            ids = [r["idx"] for r in source_rows] + [target["idx"], negative["idx"]]
                            success = {"bundle_id": f"P051-{split}-{target['idx']}-{active_key.replace(' ', '-')}",
                                       "split": split, "source_ids": ids[:-2], "positive_target_id": ids[-2], "negative_target_id": ids[-1],
                                       "product_ids": ids, "target_class": target["class"], "role_keys": role_keys,
                                       "active_role": active_index, "active_role_key": active_key, "active_family": family,
                                       "instructions": instructions(target["class"], role_keys),
                                       "valid_assignments": valid, "supervised_assignments": [True] * len(permutations),
                                       "verified_wrong_assignments": [not value for value in valid], "verified_dod_assignments": dod_pairs,
                                       "positive_semantic_energies": pos_e, "negative_semantic_energies": neg_e,
                                       "intended_assignment": list(range(len(role_keys))),
                                       "label_status": "metadata_screened_weak_labels_visual_gate_pending",
                                       "source_locations": {p: {"shard": meta[p]["shard"], "row_index": meta[p]["row_index"]} for p in ids}}
                            break
                        if success:
                            break
                    if success:
                        break
                if success:
                    chosen.append(success)
                    targets.add(target["idx"])
                    usage.update(success["product_ids"])
                    active_counts[success["active_family"]] += 1
                    role_counts[success["active_role_key"]] += 1
                    n_counts[len(success["source_ids"])] += 1
            if len(chosen) == quota:
                break
        split_reports[split] = {"groups": len(chosen), "requested_candidates": quota, "families": dict(active_counts),
                                "active_roles": dict(role_counts), "source_counts": dict(n_counts),
                                "unique_products": len(usage), "maximum_product_uses": max(usage.values(), default=0)}
        all_rows.extend(chosen)
        print(json.dumps({"split": split, **split_reports[split]}), flush=True)
    product_sets = {split: {p for row in all_rows if row["split"] == split for p in row["product_ids"]} for split in pools}
    assert not any(product_sets[a] & product_sets[b] for a, b in itertools.combinations(product_sets, 2))
    path = out / "orbits_metadata_screened.jsonl"
    with path.open("w") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {"status": "metadata_screened_visual_gate_pending", "groups": len(all_rows), "splits": split_reports,
              "product_split_overlap": 0, "excluded_historical_or_official_products": len(excluded),
              "orbits_sha256": sha(path), "metadata_sha256": manifest["sha256"],
              "visual_gate_passed": False, "pilot_authorized": False,
              "template_policy": "new metadata-derived queries; not official MERIT queries or official benchmark scores"}
    (out / "orbits_prepare_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

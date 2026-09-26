#!/usr/bin/env python3
"""重算全部轨道标签、排除图片身份泄漏，再允许视觉质检。"""
from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canon(value):
    return value.strip().lower() if isinstance(value, str) else json.dumps(value, sort_keys=True).lower()


def main():
    config = json.loads(Path("configs/P051_tracebind.json").read_text())
    run = Path("runs/P051_tracebind")
    rows = list(map(json.loads, (run / "orbits_metadata_screened.jsonl").open()))
    manifest = json.loads((run / "orbits_image_manifest.json").read_text())
    assert manifest["complete"] and manifest["orbits_sha256"] == sha(run / "orbits_metadata_screened.jsonl")
    images = {entry["idx"]: entry for entry in manifest["entries"]}
    historical_entries = {}
    for name in ["P047_position_controlled_role_contrast", "P048_visible_role_contrast"]:
        folder = Path("runs") / name
        old_rows = list(map(json.loads, (folder / "bundles.jsonl").open()))
        heldout_ids = {str(product) for row in old_rows if row["split"] != "train" for product in row["product_ids"]}
        old_manifest = json.loads((folder / "image_manifest.json").read_text())
        assert old_manifest["complete"]
        old_images = {str(entry["idx"]): entry for entry in old_manifest["entries"]}
        assert heldout_ids <= old_images.keys()
        historical_entries.update({product: old_images[product] for product in heldout_ids})
    meta = {row["idx"]: row for row in map(json.loads, Path(config["data"]["metadata_path"]).open())}
    cache_path = run / "image_pixel_hashes.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    def pixel_hash(entry):
        assert sha(entry["path"]) == entry["sha256"]
        if entry["sha256"] in cache:
            return entry["sha256"], cache[entry["sha256"]]
        with Image.open(entry["path"]) as image:
            image = image.convert("RGB")
            payload = str(image.size).encode() + image.tobytes()
        return entry["sha256"], hashlib.sha256(payload).hexdigest()
    with ThreadPoolExecutor(8) as pool:
        for raw_hash, decoded_hash in pool.map(pixel_hash, list(images.values()) + list(historical_entries.values())):
            cache[raw_hash] = decoded_hash
    cache_path.write_text(json.dumps(cache) + "\n")
    identity = {product: cache[entry["sha256"]] for product, entry in images.items()}
    historical_pixels = {cache[entry["sha256"]] for entry in historical_entries.values()}
    source_masks, dod_masks, issues = Counter(), Counter(), Counter()
    by_split = defaultdict(list)
    for row in rows:
        ids, keys = row["source_ids"], row["role_keys"]
        assert len(ids) == len(keys) in (2, 3)
        assert len(set(row["product_ids"])) == len(row["product_ids"])
        assert len(set(keys)) == len(keys)
        perms = list(itertools.permutations(range(len(ids))))
        target, negative = meta[row["positive_target_id"]], meta[row["negative_target_id"]]
        assert target["class"] == negative["class"] == row["target_class"]
        assert all(meta[source]["class"] == row["target_class"] for source in ids)
        def energy(candidate, assignment):
            return sum(canon(meta[ids[source]]["attribute"][key]) == canon(candidate["attribute"][key]) for source, key in zip(assignment, keys)) / len(keys)
        pos_e, neg_e = [energy(target, p) for p in perms], [energy(negative, p) for p in perms]
        assert pos_e[0] == 1.0 and neg_e[0] == (len(keys) - 1) / len(keys)
        assert [e == 1.0 for e in pos_e] == row["valid_assignments"]
        assert row["supervised_assignments"] == [True] * len(perms)
        assert [e < 1.0 for e in pos_e] == row["verified_wrong_assignments"]
        assert pos_e == row["positive_semantic_energies"] and neg_e == row["negative_semantic_energies"]
        assert row["verified_dod_assignments"]
        for index in row["verified_dod_assignments"]:
            assert not row["valid_assignments"][index] and pos_e[0] - pos_e[index] > neg_e[0] - neg_e[index]
        assert all(sum(f"Product {i+1}" in text for i in range(len(ids))) == len(ids) for text in row["instructions"].values())
        source_masks[len(ids)] += 1; dod_masks[len(row["verified_dod_assignments"])] += 1
        fingerprints = [identity[product] for product in row["product_ids"]]
        if set(fingerprints) & historical_pixels:
            issues["historical_heldout_pixel_alias_removed"] += 1
            continue
        if len(set(fingerprints)) != len(fingerprints):
            issues["within_group_duplicate_pixels"] += 1
            continue
        row["pixel_identities"] = fingerprints
        by_split[row["split"]].append(row)
    # 确认、验证优先保留；训练组不得接触任何被保留的 heldout 图像。
    used_heldout, output = set(), []
    for split in ["confirmation", "validation", "train"]:
        for row in sorted(by_split[split], key=lambda r: hashlib.sha256(r["bundle_id"].encode()).digest()):
            values = set(row["pixel_identities"])
            if values & used_heldout:
                issues[split + "_pixel_overlap_removed"] += 1
                continue
            output.append(row)
            if split != "train":
                used_heldout.update(values)
    path = run / "orbits_byte_disjoint.jsonl"
    with path.open("w") as f:
        for row in output:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {"status": "label_and_exact_pixel_audit_pass_visual_quality_pending", "input_groups": len(rows), "retained_groups": len(output),
              "issues_removed": dict(issues), "all_labels_recomputed": True, "source_count_distribution_before_filters": dict(source_masks),
              "dod_pair_count_distribution": dict(dod_masks), "dataset_sha256": sha(path), "image_manifest_sha256": sha(run / "orbits_image_manifest.json"),
              "split_groups": dict(Counter(row["split"] for row in output)),
              "split_families": {split: dict(Counter(row["active_family"] for row in output if row["split"] == split)) for split in by_split},
              "train_heldout_exact_pixel_overlap": 0, "heldout_between_group_exact_pixel_overlap": 0,
              "historical_heldout_exact_pixel_overlap": 0, "historical_heldout_images_checked": len(historical_entries),
              "near_duplicate_photos_not_certified_absent": True, "visual_gate_passed": False}
    (run / "orbits_integrity_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

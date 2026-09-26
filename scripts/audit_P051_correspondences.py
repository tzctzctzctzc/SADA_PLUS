#!/usr/bin/env python3
"""核验所有精细对应文件及与检索 heldout 的像素隔离。"""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pixel(path):
    with Image.open(path) as image:
        image = image.convert("RGB")
        return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def main():
    run = Path("runs/P051_tracebind")
    summary = json.loads((run / "correspondence_prepare_summary.json").read_text())
    path = run / "correspondences.jsonl"
    assert summary["status"] == "pass" and sha(path) == summary["correspondences_sha256"]
    rows = list(map(json.loads, path.open()))
    assert len(rows) == len({r["uid"] for r in rows}) == summary["pairs"]
    heldout = {identity for row in map(json.loads, (run / "orbits_byte_disjoint.jsonl").open())
               if row["split"] != "train" for identity in row["pixel_identities"]}
    def check(row):
        for kind in ["source", "target"]:
            assert sha(row[kind + "_image"]) == row[kind + "_sha256"]
        assert sha(row["point_path"]) == row["point_sha256"]
        with np.load(row["point_path"]) as points:
            source, target = points["source_yx"], points["target_yx"]
            assert source.shape == target.shape == (row["verified_points"], 2)
            assert int(points["grid_size"]) == 64
            assert np.isfinite(source).all() and np.isfinite(target).all()
            assert (source == np.floor(source)).all() and (target == np.floor(target)).all()
            assert (source >= 0).all() and (source < 64).all() and (target >= 0).all() and (target < 64).all()
            assert len(np.unique(target, axis=0)) == len(target)
        identities = [pixel(row["source_image"]), pixel(row["target_image"])]
        return {**row, "pixel_identities": identities}, bool(set(identities) & heldout)
    output, overlap = [], 0
    with ThreadPoolExecutor(8) as pool:
        for row, touches in pool.map(check, rows):
            if touches:
                overlap += 1
            else:
                output.append(row)
    out = run / "correspondences_disjoint.jsonl"
    with out.open("w") as f:
        for row in output:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {"status": "pass" if len(output) >= summary["minimum_pairs"] else "insufficient_pairs",
              "input_pairs": len(rows), "retained_pairs": len(output), "heldout_overlaps_removed": overlap,
              "all_images_and_coordinates_hash_verified": True, "coordinate_ranges_and_counts_verified": True,
              "retrieval_heldout_exact_pixel_overlap": 0, "dataset_sha256": sha(out),
              "near_duplicate_cross_dataset_overlap_not_certified_absent": True,
              "correspondence_annotations_are_pseudo_labels": True, "retrieval_role_ground_truth": False}
    (run / "correspondence_integrity_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    assert report["status"] == "pass"


if __name__ == "__main__":
    main()

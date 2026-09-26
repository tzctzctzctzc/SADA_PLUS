#!/usr/bin/env python3
"""导出分层逐图复核页；这不是独立人工标注研究。"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--passing-only", action="store_true")
    args = parser.parse_args()
    run = Path("runs/P051_tracebind")
    rows = list(map(json.loads, (run / ("pilot_orbits.jsonl" if args.passing_only else "orbits_byte_disjoint.jsonl")).open()))
    images = {entry["idx"]: entry for entry in json.loads((run / "orbits_image_manifest.json").read_text())["entries"]}
    config = json.loads(Path("configs/P051_tracebind.json").read_text())
    meta = {row["idx"]: row for row in map(json.loads, Path(config["data"]["metadata_path"]).open())}
    def canon(value):
        return value.strip().lower() if isinstance(value, str) else json.dumps(value, sort_keys=True).lower()
    eligible = []
    for row in rows:
        perms = list(itertools.permutations(range(len(row["source_ids"]))))
        key = row["active_role_key"]
        comparable = [index for index in row["verified_dod_assignments"]
                      if canon(meta[row["source_ids"][perms[index][row["active_role"]]]]["attribute"][key]) == canon(meta[row["negative_target_id"]]["attribute"][key])]
        if comparable:
            row["inspection_assignment"] = comparable[0]
            eligible.append(row)
    rows = eligible
    selected = []
    for family in ["color", "pattern", "shape", "construction"]:
        for n in [2, 3]:
            bucket = [row for row in rows if row["active_family"] == family and len(row["source_ids"]) == n]
            selected.extend(sorted(bucket, key=lambda r: hashlib.sha256(("manual-inspection" + r["bundle_id"]).encode()).digest())[:8])
    out = run / ("inspection_passing" if args.passing_only else "inspection_raw")
    out.mkdir(exist_ok=True)
    manifest = []
    for start in range(0, len(selected), 4):
        subset = selected[start:start+4]
        page = Image.new("RGB", (1200, 4 * 350), "white")
        draw = ImageDraw.Draw(page)
        for row_index, row in enumerate(subset):
            perms = list(itertools.permutations(range(len(row["source_ids"]))))
            assignment = row["inspection_assignment"]
            active = row["active_role"]
            products = [row["positive_target_id"], row["source_ids"][active], row["source_ids"][perms[assignment][active]], row["negative_target_id"]]
            labels = ["Positive target", "Intended reference", "Swapped reference", "Partial negative"]
            y = row_index * 350
            title = f"{start+row_index+1} | {row['active_family']} | {row['active_role_key']} | {len(row['source_ids'])} refs | {row['split']}"
            draw.text((10, y+5), title, fill="black")
            for col, (product, label) in enumerate(zip(products, labels)):
                entry = images[product]
                payload = Path(entry["path"]).read_bytes()
                assert hashlib.sha256(payload).hexdigest() == entry["sha256"]
                with Image.open(entry["path"]) as image:
                    tile = ImageOps.contain(image.convert("RGB"), (286, 286))
                    page.paste(tile, (col*300+(300-tile.width)//2, y+50+(286-tile.height)//2))
                draw.text((col*300+8, y+30), label, fill="black")
            manifest.append({"inspection_index": start+row_index+1, "bundle_id": row["bundle_id"], "role": row["active_role_key"], "family": row["active_family"], "product_ids": products, "page": f"page_{start//4:02d}.jpg"})
        page.save(out / f"page_{start//4:02d}.jpg", quality=95)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"inspection_groups": len(selected), "pages": (len(selected)+3)//4, "out": str(out)}), flush=True)


if __name__ == "__main__":
    main()

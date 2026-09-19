#!/usr/bin/env python3
"""Extract the frozen P046 products from fixed MERIT parquet row groups."""

from __future__ import annotations

import hashlib
import io
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from PIL import Image

ROOT = Path("/vepfs/vlms/1")
CONFIG = json.loads((ROOT / "configs/P046_self_normalized_role_corruption.json").read_text())
RUN_DIR = ROOT / "runs/P046_self_normalized_role_corruption"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extension(image_format: str | None) -> str:
    value = (image_format or "png").lower()
    return {"jpeg": "jpg", "tiff": "tif"}.get(value, value)


def extract_row_group(
    shard: int,
    row_group: int,
    wanted: dict[str, dict[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    remote = (
        f"datasets/{CONFIG['data']['dataset']}@{CONFIG['data']['dataset_revision']}/data/"
        f"train-{shard:05d}-of-{CONFIG['data']['total_shards']:05d}.parquet"
    )
    fs = HfFileSystem(endpoint="https://hf-mirror.com")
    with fs.open(remote, "rb") as handle:
        parquet = pq.ParquetFile(handle)
        table = parquet.read_row_group(row_group, columns=["idx", "image"])
    entries = []
    for offset, row in enumerate(table.to_pylist()):
        product_id = str(row["idx"])
        if product_id not in wanted:
            continue
        raw = row["image"]
        data = raw.get("bytes") if isinstance(raw, dict) else None
        if not data:
            raise RuntimeError(f"missing image bytes for {product_id}")
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            image_format = image.format
            width, height = image.size
            mode = image.mode
        expected_offset = wanted[product_id]["row_index"] - row_group * 100
        if expected_offset != offset:
            raise RuntimeError(
                f"row offset mismatch for {product_id}: expected {expected_offset}, got {offset}"
            )
        image_path = output_dir / f"{product_id}.{extension(image_format)}"
        image_path.write_bytes(data)
        entries.append(
            {
                "idx": product_id,
                "path": str(image_path.relative_to(ROOT)),
                "sha256": sha256_bytes(data),
                "bytes": len(data),
                "width": width,
                "height": height,
                "mode": mode,
                "format": image_format,
                "shard": shard,
                "row_group": row_group,
                "row_offset": offset,
            }
        )
    return entries


def main() -> None:
    assert Path.cwd().resolve() == ROOT
    bundles = [json.loads(line) for line in (RUN_DIR / "bundles.jsonl").open() if line.strip()]
    required = set()
    locations: dict[str, dict[str, Any]] = {}
    for row in bundles:
        required.update(row["product_ids"])
        required.add(row["negative_target_id"])
        locations.update(row["source_locations"])
    metadata = {
        row["idx"]: row
        for row in (
            json.loads(line)
            for line in (ROOT / CONFIG["data"]["metadata_path"]).open()
            if line.strip()
        )
    }
    for product_id in required:
        if product_id not in locations:
            locations[product_id] = {
                "shard": metadata[product_id]["shard"],
                "row_index": metadata[product_id]["row_index"],
            }

    output_dir = ROOT / "data/MERIT/P046_images"
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    cached_ids = set()
    for product_id in sorted(required):
        matches = list(output_dir.glob(f"{product_id}.*"))
        if len(matches) != 1:
            continue
        path = matches[0]
        data = path.read_bytes()
        try:
            with Image.open(io.BytesIO(data)) as image:
                image.load()
                image_format = image.format
                width, height = image.size
                mode = image.mode
        except Exception:
            continue
        location = locations[product_id]
        entries.append(
            {
                "idx": product_id,
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_bytes(data),
                "bytes": len(data),
                "width": width,
                "height": height,
                "mode": mode,
                "format": image_format,
                "shard": location["shard"],
                "row_group": location["row_index"] // 100,
                "row_offset": location["row_index"] % 100,
            }
        )
        cached_ids.add(product_id)

    grouped: dict[tuple[int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for product_id in required - cached_ids:
        location = locations[product_id]
        grouped[(location["shard"], location["row_index"] // 100)][product_id] = location

    print(json.dumps({"event": "cache_scan", "cached": len(cached_ids), "missing": len(required - cached_ids)}), flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(extract_row_group, shard, row_group, wanted, output_dir):
            (shard, row_group, len(wanted))
            for (shard, row_group), wanted in grouped.items()
        }
        for future in as_completed(futures):
            shard, row_group, expected = futures[future]
            rows = future.result()
            if len(rows) != expected:
                raise RuntimeError(
                    f"shard {shard} row group {row_group}: expected {expected}, extracted {len(rows)}"
                )
            entries.extend(rows)
            print(json.dumps({"event": "row_group", "shard": shard, "row_group": row_group, "images": len(rows)}), flush=True)

    entries.sort(key=lambda row: row["idx"])
    found = {row["idx"] for row in entries}
    manifest = {
        "status": "pass" if found == required else "fail",
        "dataset_revision": CONFIG["data"]["dataset_revision"],
        "requested_images": len(required),
        "extracted_images": len(entries),
        "row_groups_read": len(grouped),
        "complete": found == required,
        "missing": sorted(required - found),
        "entries": entries,
    }
    (RUN_DIR / "image_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in manifest.items() if key != "entries"}, indent=2))
    if not manifest["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

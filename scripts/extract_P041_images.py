#!/usr/bin/env python3
"""Extract only P041 bundle images from remote MERIT parquet row groups."""

from __future__ import annotations

import argparse
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


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extension(image_format: str | None) -> str:
    value = (image_format or "png").lower()
    return {"jpeg": "jpg", "tiff": "tif"}.get(value, value)


def extract_row_group(
    repo: str,
    revision: str,
    total_shards: int,
    shard: int,
    row_group: int,
    wanted: dict[str, dict[str, Any]],
    output_dir: Path,
    endpoint: str,
) -> list[dict[str, Any]]:
    remote = f"datasets/{repo}@{revision}/data/train-{shard:05d}-of-{total_shards:05d}.parquet"
    fs = HfFileSystem(endpoint=endpoint)
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
        suffix = extension(image_format)
        image_path = output_dir / f"{product_id}.{suffix}"
        image_path.write_bytes(data)
        expected = wanted[product_id]
        expected_offset = expected["row_index"] - row_group * 100
        if expected_offset != offset:
            raise RuntimeError(
                f"row offset mismatch for {product_id}: expected {expected_offset}, got {offset}"
            )
        entries.append(
            {
                "idx": product_id,
                "path": str(image_path),
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/P041_selective_dependence_pilot.json")
    parser.add_argument("--scope", choices=["audit", "all"], default="audit")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    run_dir = Path("runs/P041_selective_dependence")
    metadata_manifest = json.loads((run_dir / "metadata_manifest.json").read_text())
    metadata = {
        row["idx"]: row
        for row in (
            json.loads(line)
            for line in (run_dir / "products_metadata.jsonl").open()
            if line.strip()
        )
    }
    bundles = [
        json.loads(line)
        for line in (run_dir / "bundles.jsonl").open()
        if line.strip()
    ]
    if args.scope == "audit":
        audit = json.loads((run_dir / "data_contract_audit.json").read_text())
        audit_ids = {row["bundle_id"] for row in audit["rows"]}
        bundles = [row for row in bundles if row["bundle_id"] in audit_ids]

    requested_ids = sorted({value for row in bundles for value in row["product_ids"]})
    missing_metadata = [value for value in requested_ids if value not in metadata]
    if missing_metadata:
        raise RuntimeError(f"missing metadata for {len(missing_metadata)} products")

    grouped: dict[tuple[int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for product_id in requested_ids:
        row = metadata[product_id]
        grouped[(row["shard"], row["row_index"] // 100)][product_id] = row

    output_dir = Path("data/MERIT/P041_images")
    output_dir.mkdir(parents=True, exist_ok=True)
    endpoint = "https://hf-mirror.com"
    entries = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                extract_row_group,
                config["data"]["dataset"],
                metadata_manifest["dataset_revision"],
                config["data"]["selected_train_shards"]["total"],
                shard,
                row_group,
                wanted,
                output_dir,
                endpoint,
            ): (shard, row_group, len(wanted))
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
            print(
                json.dumps(
                    {
                        "event": "image_row_group",
                        "shard": shard,
                        "row_group": row_group,
                        "images": len(rows),
                    }
                ),
                flush=True,
            )

    entries.sort(key=lambda row: row["idx"])
    found = {row["idx"] for row in entries}
    missing = sorted(set(requested_ids) - found)
    manifest = {
        "scope": args.scope,
        "dataset": config["data"]["dataset"],
        "dataset_revision": metadata_manifest["dataset_revision"],
        "bundle_count": len(bundles),
        "requested_images": len(requested_ids),
        "extracted_images": len(entries),
        "row_groups_read": len(grouped),
        "complete": not missing and len(entries) == len(requested_ids),
        "missing": missing,
        "entries": entries,
    }
    manifest_path = run_dir / f"image_manifest_{args.scope}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in manifest.items() if key != "entries"}, indent=2))
    if not manifest["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()


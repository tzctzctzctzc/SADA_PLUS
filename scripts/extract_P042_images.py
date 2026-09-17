#!/usr/bin/env python3
"""Extract frozen P042 images, opening each MERIT shard only once."""

from __future__ import annotations

import hashlib
import io
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from PIL import Image

ROOT = Path("/vepfs/vlms/1")
assert Path.cwd().resolve() == ROOT
config = json.loads((ROOT / "configs/P042_role_conditioned_equivariance.json").read_text())
run_dir = ROOT / "runs/P042_role_conditioned_equivariance"
candidates = [
    json.loads(line)
    for line in (run_dir / "raw_candidates.jsonl").open()
    if line.strip()
]
assert len(candidates) == config["data"]["raw_candidate_bundles"]
metadata = {
    row["idx"]: row
    for row in (
        json.loads(line)
        for line in (ROOT / config["data"]["metadata_path"]).open()
        if line.strip()
    )
}
requested = sorted({pid for row in candidates for pid in row["product_ids"]})
assert len(requested) == 3 * len(candidates)
by_shard: dict[int, dict[int, dict[str, dict]]] = defaultdict(
    lambda: defaultdict(dict)
)
for product_id in requested:
    row = metadata[product_id]
    by_shard[row["shard"]][row["row_index"] // 100][product_id] = row

output_dir = ROOT / "data/MERIT/P042_images"
output_dir.mkdir(parents=True, exist_ok=True)


def extension(image_format: str | None) -> str:
    return {"jpeg": "jpg", "tiff": "tif"}.get(
        (image_format or "png").lower(), (image_format or "png").lower()
    )


def extract_shard(shard: int, groups: dict[int, dict[str, dict]]) -> list[dict]:
    remote = (
        f"datasets/{config['data']['dataset']}@{config['data']['dataset_revision']}"
        f"/data/train-{shard:05d}-of-{config['data']['total_dataset_shards']:05d}.parquet"
    )
    fs = HfFileSystem(endpoint="https://hf-mirror.com")
    entries = []
    with fs.open(remote, "rb") as handle:
        parquet = pq.ParquetFile(handle)
        for row_group, wanted in sorted(groups.items()):
            table = parquet.read_row_group(row_group, columns=["idx", "image"])
            found = set()
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
                expected_offset = wanted[product_id]["row_index"] - row_group * 100
                if expected_offset != offset:
                    raise RuntimeError(
                        f"row offset mismatch {product_id}: {expected_offset} != {offset}"
                    )
                entries.append(
                    {
                        "idx": product_id,
                        "path": str(image_path.relative_to(ROOT)),
                        "sha256": hashlib.sha256(data).hexdigest(),
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
                found.add(product_id)
            missing = set(wanted) - found
            if missing:
                raise RuntimeError(
                    f"shard {shard} row group {row_group} missing {sorted(missing)}"
                )
    return entries


entries = []
with ThreadPoolExecutor(max_workers=config["data"]["image_extract_workers"]) as pool:
    futures = {
        pool.submit(extract_shard, shard, groups): (
            shard,
            sum(len(wanted) for wanted in groups.values()),
            len(groups),
        )
        for shard, groups in by_shard.items()
    }
    for future in as_completed(futures):
        shard, expected, row_groups = futures[future]
        rows = future.result()
        if len(rows) != expected:
            raise RuntimeError(f"shard {shard}: expected {expected}, got {len(rows)}")
        entries.extend(rows)
        print(
            json.dumps(
                {
                    "event": "image_shard",
                    "shard": shard,
                    "row_groups": row_groups,
                    "images": len(rows),
                }
            ),
            flush=True,
        )

entries.sort(key=lambda row: row["idx"])
found = {row["idx"] for row in entries}
missing = sorted(set(requested) - found)
manifest = {
    "dataset": config["data"]["dataset"],
    "dataset_revision": config["data"]["dataset_revision"],
    "raw_candidate_bundles": len(candidates),
    "requested_images": len(requested),
    "extracted_images": len(entries),
    "shards_read": len(by_shard),
    "row_groups_read": sum(len(groups) for groups in by_shard.values()),
    "complete": not missing and found == set(requested),
    "missing": missing,
    "entries": entries,
}
(run_dir / "image_manifest_raw.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
)
print(
    json.dumps(
        {key: value for key, value in manifest.items() if key != "entries"},
        indent=2,
    )
)
if not manifest["complete"]:
    raise SystemExit(2)

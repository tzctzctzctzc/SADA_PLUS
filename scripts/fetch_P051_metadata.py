#!/usr/bin/env python3
"""只读取固定版本 MERIT 元数据列，按分片缓存，失败时可继续。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_P041 import read_one_shard


def write_jsonl(path, rows):
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/P051_tracebind.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    assert Path.cwd().resolve() == Path(config["root"])
    data = config["data"]
    out = Path(data["metadata_path"])
    shards = out.parent / "metadata_shards"
    shards.mkdir(parents=True, exist_ok=True)
    old_manifest = json.loads(Path(data["cached_metadata_manifest"]).read_text())
    assert old_manifest["dataset_revision"] == data["dataset_revision"]
    cached = {}
    for line in Path(data["cached_metadata_path"]).open():
        row = json.loads(line)
        cached.setdefault(row["shard"], []).append(row)
    for shard, rows in cached.items():
        path = shards / f"{shard:05d}.jsonl"
        if not path.exists():
            write_jsonl(path, sorted(rows, key=lambda row: row["row_index"]))

    def fetch(shard):
        path = shards / f"{shard:05d}.jsonl"
        if path.exists():
            return shard, sum(1 for _ in path.open()), "cached"
        rows = read_one_shard(data["dataset"], data["dataset_revision"], shard,
                              data["total_shards"], data["endpoint"])
        write_jsonl(path, rows)
        return shard, len(rows), "fetched"

    failures = []
    with ThreadPoolExecutor(max_workers=data["metadata_workers"]) as pool:
        futures = {pool.submit(fetch, shard): shard for shard in range(data["total_shards"])}
        for future in as_completed(futures):
            try:
                shard, count, status = future.result()
                print(json.dumps({"shard": shard, "products": count, "status": status}), flush=True)
            except Exception as exc:
                failures.append({"shard": futures[future], "error": str(exc)})
                print(json.dumps(failures[-1]), flush=True)
    if failures:
        (out.parent / "metadata_download_failures.json").write_text(json.dumps(failures, indent=2))
        raise SystemExit(2)
    ids = set()
    digest = hashlib.sha256()
    temporary = out.with_suffix(".partial")
    with temporary.open("wb") as handle:
        for shard in range(data["total_shards"]):
            for line in (shards / f"{shard:05d}.jsonl").read_bytes().splitlines(keepends=True):
                row = json.loads(line)
                assert row["idx"] not in ids, "duplicate product ID across shards"
                ids.add(row["idx"])
                handle.write(line)
                digest.update(line)
    os.replace(temporary, out)
    manifest = {"dataset": data["dataset"], "dataset_revision": data["dataset_revision"],
                "selected_shards": list(range(data["total_shards"])), "product_count": len(ids),
                "sha256": digest.hexdigest(), "complete": True}
    (out.parent / "metadata_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()

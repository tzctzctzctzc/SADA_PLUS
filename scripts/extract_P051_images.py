#!/usr/bin/env python3
"""提取反事实组的真实图片；复用旧缓存并核对实际 parquet 行位置。"""
from __future__ import annotations

import hashlib
import argparse
import io
import json
from bisect import bisect_right
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from PIL import Image


def image_entry(product, path, payload, location):
    with Image.open(io.BytesIO(payload)) as image:
        image.load(); width, height = image.size; fmt = image.format
    return {"idx": product, "path": str(path.resolve()), "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload), "width": width, "height": height, "format": fmt, **location}


def verified_cache(data, run):
    """只复用同版本清单中的 ID/字节映射，不凭任意文件名认定商品身份。"""
    paths = [run / "orbits_image_manifest.json"]
    paths += sorted(Path("runs").glob("P04*/image_manifest*.json"))
    cache = {}
    for manifest_path in paths:
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        if not manifest.get("complete") or manifest.get("dataset_revision") != data["dataset_revision"]:
            continue
        for entry in manifest["entries"]:
            path = Path(entry["path"])
            if not path.is_absolute():
                path = Path.cwd() / path
            if path.exists():
                cache.setdefault(str(entry["idx"]), (path, entry["sha256"], str(manifest_path)))
    return cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--audit-cache-provenance-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path("configs/P051_tracebind.json").read_text())
    data = config["data"]
    run, out = Path("runs/P051_tracebind"), Path("data/MERIT/P051_images")
    out.mkdir(parents=True, exist_ok=True)
    rows = list(map(json.loads, (run / "orbits_metadata_screened.jsonl").open()))
    input_sha = hashlib.sha256((run / "orbits_metadata_screened.jsonl").read_bytes()).hexdigest()
    locations = {product: row["source_locations"][product] for row in rows for product in row["product_ids"]}
    cache = verified_cache(data, run)
    if args.audit_cache_provenance_only:
        current = json.loads((run / "orbits_image_manifest.json").read_text())
        historical = {}
        for manifest_path in sorted(Path("runs").glob("P04*/image_manifest*.json")):
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("complete") and manifest.get("dataset_revision") == data["dataset_revision"]:
                for entry in manifest["entries"]:
                    historical.setdefault((str(entry["idx"]), entry["sha256"]), str(manifest_path))
        issues, counts = [], defaultdict(int)
        def check_provenance(entry):
            product = str(entry["idx"])
            assert {key: entry[key] for key in ["shard", "row_index"]} == locations[product]
            assert hashlib.sha256(Path(entry["path"]).read_bytes()).hexdigest() == entry["sha256"]
            if Path(entry["path"]).parent.name == "P051_images":
                return "new_pinned_parquet_extraction", None
            elif (product, entry["sha256"]) in historical:
                return "matching_product_bytes_in_pinned_historical_manifest", None
            else:
                return None, {"idx": product, "path": entry["path"]}
        with ThreadPoolExecutor(args.workers) as pool:
            for kind, issue in pool.map(check_provenance, current["entries"]):
                if issue:
                    issues.append(issue)
                else:
                    counts[kind] += 1
                if (sum(counts.values()) + len(issues)) % 5000 == 0:
                    print(json.dumps({"checked": sum(counts.values()) + len(issues)}), flush=True)
        report = {"status": "pass" if not issues else "unverified_legacy_cache_mapping", "counts": dict(counts), "issues": issues,
                  "dataset_revision": data["dataset_revision"], "all_current_bytes_and_metadata_locations_verified": True,
                  "scope": "new extraction trace plus pinned historical ID-to-byte manifests; not a second upstream download"}
        (run / "image_cache_provenance_audit.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        if issues:
            raise SystemExit(2)
        return
    entries, pending = [], defaultdict(dict)
    cached = [(product, cache[product], location) for product, location in locations.items() if product in cache]
    print(json.dumps({"requested": len(locations), "cached_files_to_verify": len(cached)}), flush=True)
    def verify_cached(item):
        product, (path, expected_sha, manifest_path), location = item
        payload = path.read_bytes()
        assert hashlib.sha256(payload).hexdigest() == expected_sha
        return image_entry(product, path, payload, {**location, "cache_manifest": manifest_path})
    # JPEG 解码和文件读取可并发；原串行缓存核验比网络提取还慢。
    with ThreadPoolExecutor(args.workers) as pool:
        for entry in pool.map(verify_cached, cached):
            entries.append(entry)
            if len(entries) % 1000 == 0:
                print(json.dumps({"cached_verified": len(entries), "cached_total": len(cached)}), flush=True)
    for product, location in locations.items():
        if product not in cache:
            pending[location["shard"]][product] = location
    print(json.dumps({"requested": len(locations), "cached": len(entries), "shards_to_read": len(pending)}), flush=True)

    def extract(shard, wanted):
        fs = HfFileSystem(endpoint=data["endpoint"])
        remote = f"datasets/{data['dataset']}@{data['dataset_revision']}/data/train-{shard:05d}-of-{data['total_shards']:05d}.parquet"
        results = []
        with fs.open(remote, "rb") as f:
            parquet = pq.ParquetFile(f)
            offsets = [0]
            for rg in range(parquet.num_row_groups):
                offsets.append(offsets[-1] + parquet.metadata.row_group(rg).num_rows)
            groups = defaultdict(dict)
            for product, location in wanted.items():
                rg = bisect_right(offsets, location["row_index"]) - 1
                assert rg < parquet.num_row_groups
                groups[rg][product] = location
            for rg, group in groups.items():
                table = parquet.read_row_group(rg, columns=["idx", "image"])
                found = set()
                for offset, row in enumerate(table.to_pylist()):
                    product = str(row["idx"])
                    if product not in group:
                        continue
                    assert group[product]["row_index"] == offsets[rg] + offset
                    raw = row["image"]["bytes"]
                    assert raw
                    # 后缀不决定格式；PIL 解码实际字节。
                    path = out / (product + ".img")
                    entry = image_entry(product, path, raw, group[product])
                    temporary = path.with_suffix(".partial")
                    temporary.write_bytes(raw); temporary.replace(path)
                    results.append(entry); found.add(product)
                assert found == set(group)
        return shard, results

    failures = []
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(extract, shard, wanted): shard for shard, wanted in pending.items()}
        for future in as_completed(futures):
            try:
                shard, new = future.result(); entries.extend(new)
                print(json.dumps({"shard": shard, "new_images": len(new), "total_images": len(entries)}), flush=True)
            except Exception as exc:
                failures.append({"shard": futures[future], "error": str(exc)})
                print(json.dumps(failures[-1]), flush=True)
    entries.sort(key=lambda e: e["idx"])
    assert hashlib.sha256((run / "orbits_metadata_screened.jsonl").read_bytes()).hexdigest() == input_sha
    report = {"complete": not failures and len(entries) == len(locations), "dataset_revision": data["dataset_revision"],
              "requested": len(locations), "extracted": len(entries), "failures": failures,
              "orbits_sha256": input_sha,
              "entries": entries}
    (run / "orbits_image_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "entries"}), flush=True)
    if not report["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

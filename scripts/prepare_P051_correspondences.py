#!/usr/bin/env python3
"""真实下载、校验并连接 SemAlign mask/坐标与 Subjects200K 源图。"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import requests
import torch
from PIL import Image

ENDPOINT = "https://hf-mirror.com"
SOURCE_REV = "0d1cf6536239888f1a8e218790649344810067bc"
DIR = Path("data/P051_correspondences")
RUN = Path("runs/P051_tracebind")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def api(path):
    response = requests.get(ENDPOINT + "/api/datasets/" + path, timeout=30)
    response.raise_for_status()
    return response.json()


def download(repo, revision, name, expected):
    path = DIR / name.replace("data/", "source/")
    path.parent.mkdir(parents=True, exist_ok=True)
    expected_sha = expected.get("lfs", {}).get("oid")
    if path.exists() and path.stat().st_size == expected["size"] and (not expected_sha or sha(path) == expected_sha):
        return path
    temporary = path.with_suffix(path.suffix + ".partial")
    digest = hashlib.sha256()
    with requests.get(f"{ENDPOINT}/datasets/{repo}/resolve/{revision}/{name}", stream=True, timeout=(20, 90)) as response:
        response.raise_for_status()
        with temporary.open("wb") as f:
            for chunk in response.iter_content(4 * 1048576):
                f.write(chunk); digest.update(chunk)
    assert temporary.stat().st_size == expected["size"]
    assert not expected_sha or digest.hexdigest() == expected_sha
    temporary.replace(path)
    print(json.dumps({"downloaded": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}), flush=True)
    return path


def png_bytes(image):
    buffer = io.BytesIO(); image.save(buffer, format="PNG")
    return buffer.getvalue()


def main():
    DIR.mkdir(parents=True, exist_ok=True); RUN.mkdir(parents=True, exist_ok=True)
    manifest_path = RUN / "correspondence_source_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        revision = api("ByteDance-FanQie/SemAlign-MS-Subjects200K")["sha"]
        manifest = {"annotation_revision": revision, "source_revision": SOURCE_REV,
                    "annotation_files": api(f"ByteDance-FanQie/SemAlign-MS-Subjects200K/tree/{revision}"),
                    "source_files": api(f"Yuanshi/Subjects200K/tree/{SOURCE_REV}/data"),
                    "selected_source_shards": [8, 16, 24], "minimum_pairs": 5000, "maximum_pairs": 8000}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    assert manifest["source_revision"] == SOURCE_REV
    annotations = {row["path"]: row for row in manifest["annotation_files"]}
    sources = {row["path"]: row for row in manifest["source_files"]}
    jobs = [("ByteDance-FanQie/SemAlign-MS-Subjects200K", manifest["annotation_revision"], name, annotations[name]) for name in ["coord.zip", "mask.zip"]]
    jobs += [("Yuanshi/Subjects200K", SOURCE_REV, f"data/train-{shard:05d}-of-00032.parquet", sources[f"data/train-{shard:05d}-of-00032.parquet"]) for shard in manifest["selected_source_shards"]]
    with ThreadPoolExecutor(3) as pool:
        list(pool.map(lambda args: download(*args), jobs))
    coord_zip, mask_zip = zipfile.ZipFile(DIR / "coord.zip"), zipfile.ZipFile(DIR / "mask.zip")
    coord_names = {Path(name).name: name for name in coord_zip.namelist() if name.endswith(".pt")}
    dift_names = {Path(name).name: name for name in coord_zip.namelist() if "/dift/" in "/" + name}
    geo_names = {Path(name).name: name for name in coord_zip.namelist() if "/geoaware/" in "/" + name}
    masks = {Path(name).stem: name for name in mask_zip.namelist() if name.endswith(".png")}
    print(json.dumps({"zip_masks": len(masks), "zip_dift": len(dift_names), "zip_geoaware": len(geo_names), "sample_coord_names": coord_zip.namelist()[:4]}), flush=True)
    (DIR / "images").mkdir(exist_ok=True); (DIR / "points").mkdir(exist_ok=True)
    rows, seen, counts = [], set(), {"eligible_source_rows": 0, "uid_joined": 0, "coordinate_valid": 0}
    for shard in manifest["selected_source_shards"]:
        parquet = pq.ParquetFile(DIR / f"source/train-{shard:05d}-of-00032.parquet")
        for rg in range(parquet.num_row_groups):
            for item in parquet.read_row_group(rg).to_pylist():
                quality = item.get("quality_assessment")
                if item["collection"] != "collection_2" or not quality or not all(quality.get(k, 0) >= 5 for k in ["compositeStructure", "objectConsistency", "imageQuality"]):
                    continue
                counts["eligible_source_rows"] += 1
                description = item["description"]
                image = Image.open(io.BytesIO(item["image"]["bytes"]))
                # 与固定 MOSAIC 官方加载器完全相同的裁剪与 UID 规则。
                target = image.crop((8, 8, 520, 520))
                source = image.crop((528, 8, 1040, 520))
                target_bytes, source_bytes = png_bytes(target), png_bytes(source)
                key = f"{description['item']}_{description['description_0']}_{hashlib.sha256(target_bytes).hexdigest()[:8]}"
                uid = hashlib.sha256(key.encode()).hexdigest()
                filename = uid + "_ref0.pt"
                if uid in seen or uid not in masks or filename not in dift_names or filename not in geo_names:
                    continue
                counts["uid_joined"] += 1
                dift = torch.load(io.BytesIO(coord_zip.read(dift_names[filename])), map_location="cpu", weights_only=True)
                geo = torch.load(io.BytesIO(coord_zip.read(geo_names[filename])), map_location="cpu", weights_only=True)
                assert isinstance(dift, torch.Tensor) and isinstance(geo, torch.Tensor)
                assert dift.shape == geo.shape == (4096, 2)
                mask = Image.open(io.BytesIO(mask_zip.read(masks[uid]))).convert("L").resize((64, 64), Image.Resampling.NEAREST)
                inside = torch.from_numpy(np.array(mask).reshape(-1).copy()) > 127
                keep = (dift == geo).all(-1) & inside & (dift >= 0).all(-1) & (dift < 64).all(-1)
                indices = keep.nonzero().flatten()
                if len(indices) < 32:
                    continue
                counts["coordinate_valid"] += 1
                seen.add(uid)
                target_yx = torch.stack([indices // 64, indices % 64], -1).numpy()
                source_yx = dift[indices].numpy()
                point_path = DIR / "points" / (uid + ".npz")
                np.savez_compressed(point_path, source_yx=source_yx, target_yx=target_yx, grid_size=np.array(64))
                target_path, source_path = DIR / "images" / (uid + "_target.png"), DIR / "images" / (uid + "_source.png")
                target_path.write_bytes(target_bytes); source_path.write_bytes(source_bytes)
                rows.append({"uid": uid, "source_image": str(source_path), "target_image": str(target_path),
                             "source_sha256": hashlib.sha256(source_bytes).hexdigest(), "target_sha256": hashlib.sha256(target_bytes).hexdigest(),
                             "point_path": str(point_path), "point_sha256": sha(point_path), "verified_points": len(indices),
                             "item": description["item"], "source_caption": description["description_1"], "target_caption": description["description_0"],
                             "source_shard": shard, "source_row_group": rg})
                if len(rows) % 500 == 0:
                    print(json.dumps({"pairs": len(rows), "counts": counts}), flush=True)
                if len(rows) >= manifest["maximum_pairs"]:
                    break
            if len(rows) >= manifest["maximum_pairs"]:
                break
        if len(rows) >= manifest["maximum_pairs"]:
            break
    path = RUN / "correspondences.jsonl"
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {"status": "pass" if len(rows) >= manifest["minimum_pairs"] else "insufficient_pairs",
              "pairs": len(rows), "unique_uids": len(seen), "minimum_pairs": manifest["minimum_pairs"],
              "counts": counts, "total_verified_points": sum(row["verified_points"] for row in rows),
              "correspondences_sha256": sha(path), "source_revision": SOURCE_REV,
              "annotation_revision": manifest["annotation_revision"], "target_mask_required": True,
              "dift_geoaware_agreement_required": True, "missing_annotations_never_filled": True,
              "point_grid_yx_size": 64, "retrieval_role_labels": False}
    (RUN / "correspondence_prepare_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if report["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

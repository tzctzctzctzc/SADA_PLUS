#!/usr/bin/env python3
"""Apply the frozen P043 pixel transform and select the independent audit set."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/vepfs/vlms/1")
assert Path.cwd().resolve() == ROOT
config = json.loads((ROOT / "configs/P043_role_conditioned_equivariance_maskaware_confirm.json").read_text())
run_dir = ROOT / "runs/P043_role_conditioned_equivariance_maskaware_confirm"
raw = [
    json.loads(line)
    for line in (run_dir / "raw_candidates.jsonl").open()
    if line.strip()
]
manifest = json.loads((run_dir / "image_manifest_raw.json").read_text())
assert manifest["complete"]
images = {row["idx"]: row for row in manifest["entries"]}
transform_cfg = config["transform"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transformed(product_id: str, save_path: Path | None = None) -> dict:
    entry = images[product_id]
    source_path = ROOT / entry["path"]
    assert sha256(source_path) == entry["sha256"]
    with Image.open(source_path) as image:
        rgb_image = image.convert("RGB")
    rgb = np.asarray(rgb_image, dtype=np.uint8)
    hsv_image = rgb_image.convert("HSV")
    hsv = np.asarray(hsv_image, dtype=np.uint8).copy()
    mask = (
        (hsv[:, :, 1] >= transform_cfg["saturation_threshold_uint8"])
        & (hsv[:, :, 2] >= transform_cfg["value_threshold_uint8"])
    )
    shifted = hsv.copy()
    hue = shifted[:, :, 0].astype(np.uint16)
    hue[mask] = (hue[mask] + transform_cfg["hue_offset_uint8"]) % 256
    shifted[:, :, 0] = hue.astype(np.uint8)
    changed_image = Image.fromarray(shifted, mode="HSV").convert("RGB")
    changed = np.asarray(changed_image, dtype=np.uint8)
    absolute_delta = np.abs(changed.astype(np.int16) - rgb.astype(np.int16))
    mean_delta = float(absolute_delta.mean())
    masked_mean_delta = float(absolute_delta[mask].mean()) if mask.any() else 0.0
    metrics = {
        "mask_coverage": float(mask.mean()),
        "mean_absolute_rgb_delta": mean_delta,
        "masked_mean_absolute_rgb_delta": masked_mean_delta,
        "width": rgb.shape[1],
        "height": rgb.shape[0],
        "pixel_prefilter_pass": (
            float(mask.mean()) >= transform_cfg["min_mask_coverage"]
            and masked_mean_delta >= transform_cfg["min_masked_mean_absolute_rgb_delta"]
        ),
    }
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        changed_image.save(save_path, format="PNG")
        metrics.update(
            {
                "path": str(save_path.relative_to(ROOT)),
                "sha256": sha256(save_path),
                "bytes": save_path.stat().st_size,
            }
        )
    return metrics


prefilter_rows = []
for index, row in enumerate(raw, start=1):
    role_ids = {
        "color_source": row["color_source_id"],
        "other_source": row["other_source_id"],
        "target": row["positive_target_id"],
    }
    metrics = {role: transformed(product_id) for role, product_id in role_ids.items()}
    prefilter_rows.append(
        {
            "bundle_id": row["bundle_id"],
            "pass": all(value["pixel_prefilter_pass"] for value in metrics.values()),
            "metrics": metrics,
        }
    )
    if index % 40 == 0:
        print(f"pixel metrics {index}/{len(raw)}", flush=True)

prefilter_by_id = {row["bundle_id"]: row for row in prefilter_rows}
(run_dir / "pixel_prefilter.jsonl").write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prefilter_rows)
)
selected = [
    row for row in raw if prefilter_by_id[row["bundle_id"]]["pass"]
][: config["data"]["audit_bundles"]]
if len(selected) != config["data"]["audit_bundles"]:
    summary = {
        "status": "fail",
        "raw_candidates": len(raw),
        "pixel_prefilter_passed": sum(row["pass"] for row in prefilter_rows),
        "required_audit_bundles": config["data"]["audit_bundles"],
    }
    (run_dir / "transform_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    raise SystemExit(2)

output_dir = ROOT / "data/MERIT/P043_transforms"
audit_bundles = []
for index, row in enumerate(selected, start=1):
    item = dict(row)
    paths = {}
    saved_metrics = {}
    for role, product_id in {
        "color_source": row["color_source_id"],
        "other_source": row["other_source_id"],
        "target": row["positive_target_id"],
    }.items():
        path = output_dir / f"{row['bundle_id']}_{role}.png"
        saved_metrics[role] = transformed(product_id, path)
        paths[f"{role}_transformed"] = str(path.relative_to(ROOT))
    item["transform_paths"] = paths
    item["transform_metrics"] = saved_metrics
    audit_bundles.append(item)
    if index % 20 == 0:
        print(f"saved transforms {index}/{len(selected)}", flush=True)

with (run_dir / "audit_bundles.jsonl").open("w") as handle:
    for row in audit_bundles:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
(run_dir / "pixel_prefilter.jsonl").write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prefilter_rows)
)
selected_products = [
    product_id for row in audit_bundles for product_id in row["product_ids"]
]
assert len(selected_products) == len(set(selected_products)) == 3 * len(audit_bundles)
summary = {
    "status": "pass",
    "raw_candidates": len(raw),
    "pixel_prefilter_passed": sum(row["pass"] for row in prefilter_rows),
    "audit_bundles": len(audit_bundles),
    "audit_unique_products": len(selected_products),
    "transform": transform_cfg,
    "other_key_counts": {
        key: sum(row["other_key"] == key for row in audit_bundles)
        for key in config["data"]["other_role_allowlist"]
    },
}
(run_dir / "transform_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
)
print(json.dumps(summary, ensure_ascii=False, indent=2))

"""Calibrate the Gaussian MSE action filter on source validation GT."""

import argparse
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from libs.datasets import make_dataset, truncate_feats
from libs.models.SADA import make_meta_arch
from libs.models.SADA.stage1_calibration import (
    calibrate_gaussian_mse_thresholds,
    save_calibration_artifact,
)
from libs.utils.utils import _update_config, read_yaml
from tools.audit_original_sada_pseudolabels import move_video


PERTURBATION_SEEDS = [20260730, 20260731, 20260732, 20260733]
LEVELS = [0, 1, 2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--crop-seed", type=int, default=100004)
    parser.add_argument("--quantile", type=float, default=0.80)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    random.seed(args.crop_seed)
    torch.manual_seed(args.crop_seed)
    torch.cuda.manual_seed_all(args.crop_seed)
    cfg = read_yaml(args.config)
    cfg["stage1"] = {
        "enabled": True,
        "perturbation": "gaussian_mse",
        "gaussian_std": 0.05,
        "perturbation_seeds": PERTURBATION_SEEDS,
        "levels": LEVELS,
    }
    _update_config(cfg)
    dataset = make_dataset(
        cfg["dataset_name"], False, cfg["src_val_split"], **cfg["dataset"]
    )
    model = make_meta_arch(cfg["model_name"], cfg, device=args.device).to(
        args.device
    )
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint.get("state_dict", checkpoint))
    model.eval()

    crop_rng = random.Random(args.crop_seed)
    batch_size = int(cfg.get("loader", {}).get("batch_size", 1))
    records, manifest = [], []
    with torch.no_grad():
        for batch_start in range(0, len(dataset), batch_size):
            videos = []
            for index in range(
                batch_start, min(batch_start + batch_size, len(dataset))
            ):
                item = dict(dataset[index])
                item["metadata"] = dict(item["metadata"])
                full = item.get("full_feats", item["feats"])
                item.pop("full_feats", None)
                item["feats"] = full
                item = truncate_feats(
                    item,
                    cfg["dataset"]["max_seq_len"],
                    cfg["dataset"]["trunc_thresh"],
                    cfg["dataset"].get("crop_ratio"),
                    rng=crop_rng,
                )
                item["full_feats"] = full
                videos.append(move_video(item, args.device))
                manifest.append({
                    "video_id": item["metadata"]["video_id"],
                    "crop_start": item["metadata"]["crop_start"],
                    "valid_length": item["feats"].shape[-1],
                })
            records.extend(model.collect_stage1_calibration_records(videos))

    thresholds = calibrate_gaussian_mse_thresholds(
        records,
        alpha=float(cfg["model"].get("pl_threshold", 0.7)),
        levels=LEVELS,
        quantile=args.quantile,
        num_bins=4,
    )
    metadata = {
        "split": "source_validation",
        "source_split": list(cfg["src_val_split"]),
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "crop_seed": args.crop_seed,
        "gaussian_std": 0.05,
        "perturbation_seeds": PERTURBATION_SEEDS,
        "target_gt_used": False,
    }
    save_calibration_artifact(args.output, thresholds, manifest, metadata)
    print(thresholds)


if __name__ == "__main__":
    main()

"""Measure how often source training crops truncate action boundaries."""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from libs.datasets import make_dataset
from libs.utils.utils import _update_config, read_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    _update_config(cfg)
    dataset = make_dataset(
        cfg["dataset_name"], True, cfg["src_train_split"], **cfg["dataset"]
    )
    side_patterns = {(True, True): 0, (False, True): 0,
                     (True, False): 0, (False, False): 0}
    crop_samples = left_crop_samples = right_crop_samples = 0
    retained_segments = 0
    for repeat in range(args.repeats):
        random.seed(args.seed + repeat)
        np.random.seed(args.seed + repeat)
        torch.manual_seed(args.seed + repeat)
        for index in range(len(dataset)):
            sample = dataset[index]
            original_length = int(sample["metadata"]["original_feat_len"])
            crop_start = int(sample["metadata"]["crop_start"])
            crop_end = crop_start + int(sample["feats"].shape[-1])
            left_crop = crop_start > 0
            right_crop = crop_end < original_length
            crop_samples += int(left_crop or right_crop)
            left_crop_samples += int(left_crop)
            right_crop_samples += int(right_crop)
            validity = sample["boundary_valid"].bool()
            retained_segments += int(validity.shape[0])
            for start_valid, end_valid in validity.tolist():
                side_patterns[(start_valid, end_valid)] += 1

    def ratio(value, total):
        return value / total if total else 0.0

    sample_count = len(dataset) * args.repeats
    result = {
        "audit": "source_train_crop_boundary_validity",
        "videos": len(dataset),
        "repeats": args.repeats,
        "sample_count": sample_count,
        "retained_segments": retained_segments,
        "crop_sample_rate": ratio(crop_samples, sample_count),
        "left_crop_sample_rate": ratio(left_crop_samples, sample_count),
        "right_crop_sample_rate": ratio(right_crop_samples, sample_count),
        "boundary_patterns": {
            "both_valid": side_patterns[(True, True)],
            "start_truncated": side_patterns[(False, True)],
            "end_truncated": side_patterns[(True, False)],
            "both_truncated": side_patterns[(False, False)],
        },
        "invalid_start_rate": ratio(
            side_patterns[(False, True)] + side_patterns[(False, False)],
            retained_segments,
        ),
        "invalid_end_rate": ratio(
            side_patterns[(True, False)] + side_patterns[(False, False)],
            retained_segments,
        ),
        "any_invalid_rate": ratio(
            retained_segments - side_patterns[(True, True)], retained_segments
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

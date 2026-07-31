"""Overfit the frozen-base BEGTR head on one uncropped source video."""

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
from libs.models.SADA import make_meta_arch
from libs.models.SADA.boundary_target_utils import (
    boundary_focal_loss,
    build_boundary_heatmaps,
)
from libs.utils.utils import _update_config, read_yaml


def load_model(config_path, checkpoint_path, device, model_kind):
    cfg = read_yaml(config_path)
    cfg["stage1"] = {"enabled": False, "perturbation": "shift"}
    cfg["diagnostic"] = {
        "oracle_routing": "off",
        "oracle_target_regression": "off",
        "target_regression_levels": [0, 1, 2],
    }
    cfg.setdefault("begtr", {})["mode"] = "source"
    _update_config(cfg)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = make_meta_arch(cfg["model_name"], cfg, device=device).to(device)
    state_key = "ema_state_dict" if model_kind == "ema" else "state_dict"
    base_state = {
        key: value for key, value in checkpoint[state_key].items()
        if not key.startswith("aux_boundary_head.")
    }
    incompatible = model.load_state_dict(base_state, strict=False)
    allowed = {
        name for name in model.state_dict()
        if name.startswith("aux_boundary_head.")
    }
    if incompatible.unexpected_keys or not set(incompatible.missing_keys).issubset(allowed):
        raise RuntimeError(
            "checkpoint mismatch: missing={} unexpected={}".format(
                incompatible.missing_keys, incompatible.unexpected_keys
            )
        )
    model.eval()
    return cfg, checkpoint, model


@torch.no_grad()
def encode_source(model, video):
    inputs, masks = model.preprocessing([video], mode="validation")
    features, feature_masks, _ = model.backbone_source(inputs, masks)
    fpn_features, fpn_masks = model.neck_source(features, feature_masks)
    return (
        [feature.detach() for feature in fpn_features],
        [mask.detach() for mask in fpn_masks],
        model.point_generator(fpn_features),
    )


@torch.no_grad()
def peak_metrics(model, video, features, masks, points):
    starts, ends = model.aux_boundary_head(features, masks)
    start_errors, end_errors = [], []
    for level, (start, end, point) in enumerate(zip(starts, ends, points)):
        valid = masks[level][0, 0]
        centers = point[:, 0][valid]
        stride = float(point[0, 3])
        low, high = map(float, model.reg_range[level])
        start_probability = start[0][valid].sigmoid()
        end_probability = end[0][valid].sigmoid()
        for segment in video["segments"]:
            duration = float(segment[1] - segment[0])
            scale = duration / 2.0
            if duration <= 0 or scale < low or scale > high:
                continue
            radius = max(0.25 * duration, 2.0 * stride)
            for boundary, probability, output in (
                (float(segment[0]), start_probability, start_errors),
                (float(segment[1]), end_probability, end_errors),
            ):
                local = torch.where(
                    (centers >= boundary - radius)
                    & (centers <= boundary + radius)
                )[0]
                if local.numel():
                    peak = centers[local[probability[local].argmax()]]
                    output.append(abs(float(peak) - boundary) / duration)
    return {
        "start_count": len(start_errors),
        "end_count": len(end_errors),
        "start_normalized_error": float(np.mean(start_errors)),
        "end_normalized_error": float(np.mean(end_errors)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-kind", choices=("ema", "student"), default="ema")
    parser.add_argument("--video-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cfg, checkpoint, model = load_model(
        args.config, args.checkpoint, args.device, args.model_kind
    )
    dataset = make_dataset(
        cfg["dataset_name"], False, cfg["src_val_split"], **cfg["dataset"]
    )
    raw_video = dataset[args.video_index]
    if raw_video["segments"] is None or not raw_video["segments"].numel():
        raise ValueError("selected source video has no actions")
    video = {
        key: value.to(args.device) if isinstance(value, torch.Tensor) else value
        for key, value in raw_video.items()
    }
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.aux_boundary_head.parameters():
        parameter.requires_grad_(True)
    features, masks, points = encode_source(model, video)
    targets = build_boundary_heatmaps(
        points, masks, [video["segments"]], model.reg_range,
        sigma=model.begtr_sigma,
    )
    optimizer = torch.optim.AdamW(
        model.aux_boundary_head.parameters(), lr=args.lr, weight_decay=1e-3
    )

    def current_loss():
        starts, ends = model.aux_boundary_head(features, masks)
        return boundary_focal_loss(
            starts, ends, targets[0], targets[1], targets[2], targets[3]
        )

    before_loss = float(current_loss().detach())
    before_peak = peak_metrics(model, video, features, masks, points)
    best_loss = before_loss
    best_state = None
    history = []
    for step in range(args.steps):
        learning_rate = args.lr * min(
            1.0, (step + 1) / max(args.warmup_steps, 1)
        )
        optimizer.param_groups[0]["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        loss = current_loss()
        loss.backward()
        optimizer.step()
        value = float(loss.detach())
        if value < best_loss:
            best_loss = value
            best_state = {
                key: tensor.detach().cpu().clone()
                for key, tensor in model.aux_boundary_head.state_dict().items()
            }
        if step in {0, args.warmup_steps - 1, args.steps - 1}:
            history.append({"step": step + 1, "loss": value, "lr": learning_rate})
    if best_state is not None:
        model.aux_boundary_head.load_state_dict(best_state)
    after_loss = float(current_loss().detach())
    after_peak = peak_metrics(model, video, features, masks, points)
    base_gradients = sum(
        parameter.grad is not None
        for name, parameter in model.named_parameters()
        if not name.startswith("aux_boundary_head.")
    )
    result = {
        "audit": "single_uncropped_source_video_overfit",
        "checkpoint_epoch": checkpoint.get("epoch"),
        "video_id": raw_video["metadata"]["video_id"],
        "segments": int(raw_video["segments"].shape[0]),
        "steps": args.steps,
        "learning_rate": args.lr,
        "warmup_steps": args.warmup_steps,
        "loss": {"before": before_loss, "best": best_loss, "after": after_loss},
        "peak": {"before": before_peak, "after": after_peak},
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.aux_boundary_head.parameters()
        ),
        "base_parameters_with_gradients": base_gradients,
        "history": history,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

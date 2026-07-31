"""Train only the BEGTR auxiliary head with source boundary supervision."""

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from libs.datasets import make_dataset
from libs.datasets.data_utils import trivial_batch_collator, worker_init_reset_seed
from libs.models.SADA import make_meta_arch
from libs.models.SADA.boundary_target_utils import (
    boundary_focal_loss,
    build_boundary_heatmaps,
)
from libs.utils.utils import _update_config, read_yaml


def move_batch(batch, device):
    return [{
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in video.items()
    } for video in batch]


def load_model(cfg, checkpoint_path, device, model_kind):
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
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.aux_boundary_head.parameters():
        parameter.requires_grad_(True)
    model.eval()
    return checkpoint, model


@torch.no_grad()
def encode(model, batch, mode):
    inputs, masks = model.preprocessing(batch, mode=mode)
    features, feature_masks, _ = model.backbone_source(inputs, masks)
    fpn_features, fpn_masks = model.neck_source(features, feature_masks)
    return (
        [feature.detach() for feature in fpn_features],
        [mask.detach() for mask in fpn_masks],
        model.point_generator(fpn_features),
    )


def make_targets(model, batch, masks, points):
    boundary_valid = [video["boundary_valid"] for video in batch]
    crop_edges = [torch.tensor((
        int(video["metadata"].get("crop_start", 0)) > 0,
        int(video["metadata"].get("crop_start", 0))
        + int(video["feats"].shape[-1])
        < int(video["metadata"].get(
            "original_feat_len", video["feats"].shape[-1]
        )),
    ), dtype=torch.bool, device=model.device) for video in batch]
    return build_boundary_heatmaps(
        points,
        masks,
        [video["segments"] for video in batch],
        model.reg_range,
        sigma=model.begtr_sigma,
        gt_boundary_valid=boundary_valid,
        crop_edges=crop_edges,
        edge_ignore_cells=model.begtr_edge_ignore_cells,
    )


def batch_loss(model, batch, mode):
    features, masks, points = encode(model, batch, mode)
    targets = make_targets(model, batch, masks, points)
    starts, ends = model.aux_boundary_head(features, masks)
    loss = boundary_focal_loss(
        starts, ends, targets[0], targets[1], targets[2], targets[3]
    )
    return loss, (starts, ends, masks, points)


@torch.no_grad()
def validate(model, dataset, device):
    losses, start_errors, end_errors = [], [], []
    model.aux_boundary_head.eval()
    for index in range(len(dataset)):
        batch = move_batch([dataset[index]], device)
        loss, outputs = batch_loss(model, batch, "validation")
        losses.append(float(loss))
        starts, ends, masks, points = outputs
        video = batch[0]
        for level, (start, end, point) in enumerate(zip(starts, ends, points)):
            valid = masks[level][0, 0]
            centers = point[:, 0][valid]
            stride = float(point[0, 3])
            low, high = map(float, model.reg_range[level])
            probabilities = (start[0][valid].sigmoid(), end[0][valid].sigmoid())
            for segment in video["segments"]:
                duration = float(segment[1] - segment[0])
                scale = duration / 2.0
                if duration <= 0 or scale < low or scale > high:
                    continue
                radius = max(0.25 * duration, 2.0 * stride)
                for boundary, probability, errors in (
                    (float(segment[0]), probabilities[0], start_errors),
                    (float(segment[1]), probabilities[1], end_errors),
                ):
                    local = torch.where(
                        (centers >= boundary - radius)
                        & (centers <= boundary + radius)
                    )[0]
                    if local.numel():
                        peak = centers[local[probability[local].argmax()]]
                        errors.append(abs(float(peak) - boundary) / duration)
    return {
        "loss": float(np.mean(losses)),
        "start_normalized_error": float(np.mean(start_errors)),
        "end_normalized_error": float(np.mean(end_errors)),
        "boundary_count": len(start_errors) + len(end_errors),
    }


def learning_rate(step, total_steps, warmup_steps, maximum, minimum_ratio):
    if step < warmup_steps:
        return maximum * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return maximum * (minimum_ratio + (1.0 - minimum_ratio) * cosine)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-kind", choices=("ema", "student"), default="ema")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--min-lr-ratio", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cfg = read_yaml(args.config)
    cfg["stage1"] = {"enabled": False, "perturbation": "shift"}
    cfg["diagnostic"] = {
        "oracle_routing": "off",
        "oracle_target_regression": "off",
        "target_regression_levels": [0, 1, 2],
    }
    cfg.setdefault("begtr", {}).update({
        "mode": "source",
        "edge_ignore_cells": int(cfg.get("begtr", {}).get(
            "edge_ignore_cells", 3
        )),
    })
    _update_config(cfg)
    checkpoint, model = load_model(
        cfg, args.checkpoint, args.device, args.model_kind
    )
    train_dataset = make_dataset(
        cfg["dataset_name"], True, cfg["src_train_split"], **cfg["dataset"]
    )
    val_dataset = make_dataset(
        cfg["dataset_name"], False, cfg["src_val_split"], **cfg["dataset"]
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=trivial_batch_collator,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_reset_seed if args.num_workers else None,
        persistent_workers=args.num_workers > 0,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        model.aux_boundary_head.parameters(), lr=args.lr,
        weight_decay=args.weight_decay,
    )
    total_steps = args.epochs * len(loader)
    warmup_steps = args.warmup_epochs * len(loader)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = output_dir / "config.json"
    resolved_config.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    best_loss = float("inf")
    stale_epochs = 0
    global_step = 0
    history = []
    best_checkpoint = output_dir / "best.pt"
    for epoch in range(1, args.epochs + 1):
        model.aux_boundary_head.train()
        train_losses = []
        for raw_batch in loader:
            batch = move_batch(raw_batch, args.device)
            lr = learning_rate(
                global_step, total_steps, warmup_steps,
                args.lr, args.min_lr_ratio,
            )
            optimizer.param_groups[0]["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            loss, _ = batch_loss(model, batch, "train")
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach()))
            global_step += 1
        metrics = validate(model, val_dataset, args.device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "source_val": metrics,
        }
        history.append(row)
        print(json.dumps(row))
        if metrics["loss"] < best_loss - args.min_delta:
            best_loss = metrics["loss"]
            stale_epochs = 0
            state = model.state_dict()
            torch.save({
                "epoch": epoch,
                "state_dict": state,
                "ema_state_dict": state,
                "config": cfg,
                "source_checkpoint": args.checkpoint,
                "source_checkpoint_epoch": checkpoint.get("epoch"),
                "head_only": True,
            }, best_checkpoint)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break
    result = {
        "training": "source_only_frozen_base_aux_boundary_head",
        "source_checkpoint_epoch": checkpoint.get("epoch"),
        "best_source_val_loss": best_loss,
        "best_checkpoint": str(best_checkpoint),
        "resolved_config": str(resolved_config),
        "epochs_completed": len(history),
        "early_stopped": len(history) < args.epochs,
        "history": history,
        "target_data_usage": "none",
    }
    (output_dir / "training.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Source-calibrated frozen-teacher boundary residual correction audit."""

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
from libs.models.SADA.stage2_alqa import regression_hidden_and_offsets
from libs.models.SADA.stage2_utils import aligned_boundary_iou_from_offsets
from libs.utils.utils import _update_config, read_yaml
from tools.audit_original_sada_pseudolabels import crop_targets, move_video


def parse_numbers(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def current_logits(model, inputs, masks, domain):
    if domain == "source":
        backbone, neck, head = (
            model.backbone_source, model.neck_source, model.cls_head
        )
    else:
        backbone, neck, head = (
            model.backbone_target, model.neck_target, model.cls_head_target
        )
    features, feature_masks, _ = backbone(inputs, masks)
    fpn_features, fpn_masks = neck(features, feature_masks)
    logits, _ = head(fpn_features, fpn_masks)
    return torch.cat([value.permute(0, 2, 1) for value in logits], dim=1)


def collect_records(model, dataset, annotations, cfg, domain, split, levels, device):
    records = {key: [] for key in ("hidden", "offsets", "gt_offsets", "selected")}
    alpha = float(cfg["model"].get("pl_threshold", 0.7))
    batch_size = int(cfg.get("loader", {}).get("batch_size", 1))
    preprocessing_mode = "train" if split == "train" else "validation"
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            raw_batch = [
                dataset[index]
                for index in range(start, min(start + batch_size, len(dataset)))
            ]
            batch = [move_video(video, device) for video in raw_batch]
            inputs, masks = model.preprocessing(batch, mode=preprocessing_mode)
            logits = current_logits(model, inputs, masks, domain)
            teacher_features, teacher_masks, _ = model.backbone_pl(inputs, masks)
            teacher_fpn, teacher_fpn_masks = model.neck_pl(
                teacher_features, teacher_masks
            )
            hidden, offsets = regression_hidden_and_offsets(
                model.reg_head_pl, teacher_fpn, teacher_fpn_masks
            )
            points = model.point_generator(teacher_fpn)
            gt_segments, gt_labels = [], []
            for raw_video, video in zip(raw_batch, batch):
                segments, labels = crop_targets(
                    annotations[raw_video["metadata"]["video_id"]],
                    video["metadata"],
                    video["feats"].shape[-1],
                    cfg,
                    device,
                )
                gt_segments.append(segments)
                gt_labels.append(labels)
            gt_cls, gt_offsets = model.label_points(points, gt_segments, gt_labels)
            gt_cls = torch.stack(gt_cls)
            gt_offsets = torch.stack(gt_offsets)
            confidence, predicted_class = logits.sigmoid().max(dim=-1)
            offset = 0
            for level, point in enumerate(points):
                next_offset = offset + point.shape[0]
                if level not in levels:
                    offset = next_offset
                    continue
                valid = teacher_fpn_masks[level].squeeze(1)
                positive = gt_cls[:, offset:next_offset].sum(dim=-1) > 0
                class_correct = gt_cls[:, offset:next_offset].gather(
                    -1, predicted_class[:, offset:next_offset].unsqueeze(-1)
                ).squeeze(-1) > 0
                selected = valid & (confidence[:, offset:next_offset] >= alpha)
                if domain == "source":
                    selected = selected & positive & class_correct
                if selected.any():
                    records["hidden"].append(
                        hidden[level].permute(0, 2, 1)[selected].detach().cpu()
                    )
                    records["offsets"].append(
                        offsets[level].permute(0, 2, 1)[selected].detach().cpu()
                    )
                    records["gt_offsets"].append(
                        gt_offsets[:, offset:next_offset][selected].detach().cpu()
                    )
                    records["selected"].append(
                        torch.ones(int(selected.sum()), dtype=torch.bool)
                    )
                offset = next_offset
    if not records["hidden"]:
        raise RuntimeError("No selected anchors")
    return {key: torch.cat(value) for key, value in records.items()}


def fit_ridge(features, residuals, l2, device):
    features = features.to(device, dtype=torch.float64)
    residuals = residuals.to(device, dtype=torch.float64)
    mean = features.mean(dim=0)
    std = features.std(dim=0, unbiased=False).clamp_min(1e-6)
    normalized = (features - mean) / std
    augmented = torch.cat((normalized, torch.ones_like(normalized[:, :1])), dim=1)
    identity = torch.eye(augmented.shape[1], device=device, dtype=torch.float64)
    identity[-1, -1] = 0.0
    weight = torch.linalg.solve(
        augmented.T.matmul(augmented) + l2 * identity,
        augmented.T.matmul(residuals),
    )
    return {"mean": mean.cpu(), "std": std.cpu(), "weight": weight.cpu()}


def corrected_offsets(features, offsets, calibration, gamma, device):
    features = features.to(device, dtype=torch.float64)
    offsets = offsets.to(device, dtype=torch.float64)
    normalized = (features - calibration["mean"].to(device)) / calibration["std"].to(device)
    augmented = torch.cat((normalized, torch.ones_like(normalized[:, :1])), dim=1)
    correction = augmented.matmul(calibration["weight"].to(device))
    return (offsets + gamma * correction).clamp_min(0).float()


def metrics(offsets, targets):
    iou = aligned_boundary_iou_from_offsets(offsets, targets)
    return {
        "anchors": int(iou.numel()),
        "mean_boundary_iou": float(iou.mean()),
        "iou50_rate": float((iou >= 0.5).float().mean()),
        "offset_mse": float(torch.mean((offsets - targets).square())),
    }


def source_select(args):
    seed_everything(args.seed)
    cfg = read_yaml(args.config)
    cfg["stage1"] = {"enabled": False, "perturbation": "shift"}
    _update_config(cfg)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model = make_meta_arch(cfg["model_name"], cfg, device=args.device).to(args.device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    annotations = json.loads(
        Path(cfg["dataset"]["json_file"]).read_text(encoding="utf-8")
    )["database"]
    levels = [int(value) for value in args.levels.split(",")]
    train = make_dataset(cfg["dataset_name"], True, cfg["src_train_split"], **cfg["dataset"])
    val = make_dataset(cfg["dataset_name"], False, cfg["src_val_split"], **cfg["dataset"])
    train_records = collect_records(model, train, annotations, cfg, "source", "train", levels, args.device)
    val_records = collect_records(model, val, annotations, cfg, "source", "val", levels, args.device)
    baseline = metrics(val_records["offsets"], val_records["gt_offsets"])
    candidates = []
    best = None
    for l2 in parse_numbers(args.ridge_l2):
        calibration = fit_ridge(
            train_records["hidden"],
            train_records["gt_offsets"] - train_records["offsets"],
            l2,
            args.device,
        )
        for gamma in parse_numbers(args.correction_scales):
            corrected = corrected_offsets(
                val_records["hidden"], val_records["offsets"], calibration, gamma,
                args.device,
            ).cpu()
            result = {"l2": l2, "gamma": gamma, "metrics": metrics(corrected, val_records["gt_offsets"])}
            candidates.append(result)
            score = result["metrics"]["mean_boundary_iou"]
            if best is None or score > best["metrics"]["mean_boundary_iou"]:
                best = result
                best_calibration = calibration
    artifact = {
        "calibration": best_calibration,
        "l2": best["l2"],
        "gamma": best["gamma"],
        "config": args.config,
        "checkpoint": args.checkpoint,
        "levels": levels,
    }
    torch.save(artifact, args.calibration_output)
    return {
        "audit": "source_only_signed_boundary_residual_selection",
        "checkpoint_epoch": checkpoint.get("epoch"),
        "train_anchors": int(train_records["hidden"].shape[0]),
        "source_val_baseline": baseline,
        "candidates": candidates,
        "selected": best,
        "calibration_output": args.calibration_output,
        "target_gt_usage": "none",
    }


def target_eval(args):
    seed_everything(args.seed)
    artifact = torch.load(args.calibration, map_location="cpu")
    cfg = read_yaml(args.config)
    cfg["stage1"] = {"enabled": False, "perturbation": "shift"}
    _update_config(cfg)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model = make_meta_arch(cfg["model_name"], cfg, device=args.device).to(args.device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    annotations = json.loads(
        Path(cfg["dataset"]["json_file"]).read_text(encoding="utf-8")
    )["database"]
    val = make_dataset(cfg["dataset_name"], False, cfg["tgt_val_split"], **cfg["dataset"])
    records = collect_records(model, val, annotations, cfg, "target", "val", artifact["levels"], args.device)
    corrected = corrected_offsets(
        records["hidden"], records["offsets"], artifact["calibration"],
        artifact["gamma"], args.device,
    ).cpu()
    return {
        "audit": "fixed_source_only_signed_boundary_residual_target_evaluation",
        "checkpoint_epoch": checkpoint.get("epoch"),
        "fixed_source_selection": {"l2": artifact["l2"], "gamma": artifact["gamma"]},
        "target_baseline": metrics(records["offsets"], records["gt_offsets"]),
        "target_corrected": metrics(corrected, records["gt_offsets"]),
        "target_gt_usage": "offline evaluation only",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("source_select", "target_eval"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--calibration-output")
    parser.add_argument("--calibration")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--levels", default="0,1,2")
    parser.add_argument("--ridge-l2", default="0.1,1,10,100")
    parser.add_argument("--correction-scales", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args()
    if args.mode == "source_select" and not args.calibration_output:
        parser.error("--calibration-output is required for source_select")
    if args.mode == "target_eval" and not args.calibration:
        parser.error("--calibration is required for target_eval")
    result = source_select(args) if args.mode == "source_select" else target_eval(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Audit whether class contrast across predicted boundaries ranks offset error."""

import argparse
import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from libs.datasets import make_dataset
from libs.models.SADA import make_meta_arch
from libs.utils.utils import _update_config, read_yaml
from tools.audit_original_sada_pseudolabels import crop_targets, move_video


def parse_numbers(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_outputs(model, inputs, masks, domain):
    if domain == "source":
        backbone, neck, cls_head = (
            model.backbone_source, model.neck_source, model.cls_head,
        )
    else:
        backbone, neck, cls_head = (
            model.backbone_target, model.neck_target, model.cls_head_target,
        )
    features, feature_masks, _ = backbone(inputs, masks)
    fpn_features, fpn_masks = neck(features, feature_masks)
    logits, _ = cls_head(fpn_features, fpn_masks)
    teacher_features, teacher_masks, _ = model.backbone_pl(inputs, masks)
    teacher_fpn, teacher_fpn_masks = model.neck_pl(
        teacher_features, teacher_masks,
    )
    offsets, _ = model.reg_head_pl(teacher_fpn, teacher_fpn_masks)
    return logits, offsets, teacher_fpn, teacher_fpn_masks


def boundary_contrast(probabilities, offsets, point_grid):
    """Per-anchor foreground-vs-outside class contrast at both boundaries."""
    grid, stride = point_grid[:, 0], point_grid[:, 3]
    confidence, predicted_class = probabilities.max(dim=-1)
    starts = grid[None, :] - offsets[:, 0] * stride[None, :]
    ends = grid[None, :] + offsets[:, 1] * stride[None, :]
    left = torch.searchsorted(grid.contiguous(), (starts - stride[None, :]).contiguous())
    right = torch.searchsorted(grid.contiguous(), (ends + stride[None, :]).contiguous())
    left = left.clamp(0, grid.numel() - 1)
    right = right.clamp(0, grid.numel() - 1)
    classes = predicted_class.unsqueeze(-1)
    left_prob = probabilities.gather(1, left.unsqueeze(-1).expand_as(probabilities)).gather(
        -1, classes,
    ).squeeze(-1)
    right_prob = probabilities.gather(1, right.unsqueeze(-1).expand_as(probabilities)).gather(
        -1, classes,
    ).squeeze(-1)
    return confidence, predicted_class, (
        confidence - 0.5 * (left_prob + right_prob)
    ).clamp_min(0.0)


def collect_records(model, dataset, annotations, cfg, domain, split, levels, device):
    records = {
        key: [] for key in ("confidence", "contrast", "error", "sample_ids")
    }
    alpha = float(cfg["model"].get("pl_threshold", 0.7))
    batch_size = int(cfg.get("loader", {}).get("batch_size", 1))
    mode = "train" if split == "train" else "validation"
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            raw_batch = [
                dataset[index]
                for index in range(start, min(start + batch_size, len(dataset)))
            ]
            batch = [move_video(video, device) for video in raw_batch]
            inputs, masks = model.preprocessing(batch, mode=mode)
            logits, offsets, teacher_fpn, teacher_masks = model_outputs(
                model, inputs, masks, domain,
            )
            points = model.point_generator(teacher_fpn)
            gt_segments, gt_labels = [], []
            for raw_video, video in zip(raw_batch, batch):
                segments, labels = crop_targets(
                    annotations[raw_video["metadata"]["video_id"]],
                    video["metadata"], video["feats"].shape[-1], cfg, device,
                )
                gt_segments.append(segments)
                gt_labels.append(labels)
            gt_cls, gt_offsets = model.label_points(points, gt_segments, gt_labels)
            gt_cls = torch.stack(gt_cls)
            gt_offsets = torch.stack(gt_offsets)
            offset = 0
            for level, point in enumerate(points):
                next_offset = offset + point.shape[0]
                if level not in levels:
                    offset = next_offset
                    continue
                probabilities = logits[level].permute(0, 2, 1).sigmoid()
                confidence, predicted_class, contrast = boundary_contrast(
                    probabilities, offsets[level], point,
                )
                valid = teacher_masks[level].squeeze(1)
                selected = valid & (confidence >= alpha)
                if domain == "source":
                    labels = gt_cls[:, offset:next_offset]
                    selected &= labels.sum(dim=-1) > 0
                    selected &= labels.gather(
                        -1, predicted_class.unsqueeze(-1),
                    ).squeeze(-1) > 0
                if selected.any():
                    records["confidence"].append(confidence[selected].cpu())
                    records["contrast"].append(contrast[selected].cpu())
                    records["error"].append(
                        (offsets[level].permute(0, 2, 1)
                         - gt_offsets[:, offset:next_offset])
                        .square().mean(dim=-1)[selected].cpu()
                    )
                    sample_ids = torch.arange(
                        start, start + len(batch), device=device,
                    )[:, None].expand_as(selected)
                    records["sample_ids"].append(sample_ids[selected].cpu())
                offset = next_offset
    if not records["error"]:
        raise RuntimeError("No selected anchors")
    return {key: torch.cat(value) for key, value in records.items()}


def candidate_metrics(records, power, fraction):
    scores = records["contrast"] * records["confidence"].pow(power)
    gate = torch.zeros_like(scores)
    for sample_id in records["sample_ids"].unique(sorted=True):
        sample_indices = torch.where(records["sample_ids"] == sample_id)[0]
        keep = max(1, int(torch.ceil(torch.tensor(
            fraction * sample_indices.numel(), device=scores.device,
        )).item()))
        order = torch.argsort(scores[sample_indices], descending=True)
        gate[sample_indices[order[:keep]]] = 1.0
    weights = scores * gate
    selected = torch.where(gate > 0)[0]
    return {
        "weighted_offset_mse": float(
            (weights * records["error"]).sum() / weights.sum().clamp_min(1e-6)
        ),
        "unweighted_offset_mse": float(records["error"].mean()),
        "kept_fraction": float(gate.mean()),
        "contrast_mean": float(records["contrast"].mean()),
        "kept_contrast_mean": float(records["contrast"][selected].mean()),
    }


def load_model(args):
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
    return cfg, checkpoint, model, annotations


def source_select(args):
    seed_everything(args.seed)
    cfg, checkpoint, model, annotations = load_model(args)
    levels = [int(value) for value in args.levels.split(",")]
    train = make_dataset(cfg["dataset_name"], True, cfg["src_train_split"], **cfg["dataset"])
    val = make_dataset(cfg["dataset_name"], False, cfg["src_val_split"], **cfg["dataset"])
    train_records = collect_records(model, train, annotations, cfg, "source", "train", levels, args.device)
    val_records = collect_records(model, val, annotations, cfg, "source", "val", levels, args.device)
    candidates, best = [], None
    for power in parse_numbers(args.confidence_powers):
        for fraction in parse_numbers(args.keep_fractions):
            candidate = {
                "power": power, "fraction": fraction,
                "metrics": candidate_metrics(val_records, power, fraction),
            }
            candidates.append(candidate)
            if best is None or (
                candidate["metrics"]["weighted_offset_mse"]
                < best["metrics"]["weighted_offset_mse"]
            ):
                best = candidate
    artifact = {
        "power": best["power"], "fraction": best["fraction"],
        "levels": levels, "config": args.config, "checkpoint": args.checkpoint,
    }
    torch.save(artifact, args.calibration_output)
    return {
        "audit": "source_only_semantic_boundary_contrast_selection",
        "checkpoint_epoch": checkpoint.get("epoch"),
        "source_train_anchors": int(train_records["error"].numel()),
        "source_val_anchors": int(val_records["error"].numel()),
        "candidates": candidates, "selected": best,
        "selection_metric": "source_val_weighted_offset_mse",
        "calibration_output": args.calibration_output,
        "target_gt_usage": "none",
    }


def target_eval(args):
    artifact = torch.load(args.calibration, map_location="cpu")
    seed_everything(args.seed)
    cfg, checkpoint, model, annotations = load_model(args)
    val = make_dataset(cfg["dataset_name"], False, cfg["tgt_val_split"], **cfg["dataset"])
    records = collect_records(
        model, val, annotations, cfg, "target", "val", artifact["levels"], args.device,
    )
    return {
        "audit": "fixed_source_only_semantic_boundary_contrast_target_evaluation",
        "checkpoint_epoch": checkpoint.get("epoch"),
        "fixed_source_selection": artifact,
        "target_metrics": candidate_metrics(records, artifact["power"], artifact["fraction"]),
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
    parser.add_argument("--confidence-powers", default="0,0.5,1")
    parser.add_argument("--keep-fractions", default="1,0.75,0.5")
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

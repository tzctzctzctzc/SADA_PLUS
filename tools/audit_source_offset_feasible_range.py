"""Audit source-GT calibrated offset ranges as a target boundary-error signal.

The source split is the only calibration/selection source.  Target labels are
read exclusively by ``target_eval`` after the calibration is fixed.
"""

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
    offsets, _ = model.reg_head_target(fpn_features, fpn_masks)
    return logits, offsets, fpn_masks


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


def dataset_for(cfg, domain, split):
    is_train = split == "train"
    if domain == "source":
        split_name = cfg["src_train_split"] if is_train else cfg["src_val_split"]
    else:
        split_name = cfg["tgt_train_split"] if is_train else cfg["tgt_val_split"]
    return make_dataset(cfg["dataset_name"], is_train, split_name, **cfg["dataset"])


def collect_source_gt(model, dataset, annotations, cfg, levels, device):
    values = {}
    batch_size = int(cfg.get("loader", {}).get("batch_size", 1))
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            raw_batch = [
                dataset[index]
                for index in range(start, min(start + batch_size, len(dataset)))
            ]
            batch = [move_video(video, device) for video in raw_batch]
            inputs, masks = model.preprocessing(batch, mode="train")
            _, offsets, fpn_masks = model_outputs(model, inputs, masks, "source")
            points = model.point_generator(offsets)
            segments, labels = [], []
            for raw_video, video in zip(raw_batch, batch):
                segment, label = crop_targets(
                    annotations[raw_video["metadata"]["video_id"]],
                    video["metadata"], video["feats"].shape[-1], cfg, device,
                )
                segments.append(segment)
                labels.append(label)
            gt_cls, gt_offsets = model.label_points(points, segments, labels)
            gt_cls, gt_offsets = torch.stack(gt_cls), torch.stack(gt_offsets)
            point_start = 0
            for level, point in enumerate(points):
                point_end = point_start + point.shape[0]
                if level not in levels:
                    point_start = point_end
                    continue
                valid = fpn_masks[level].squeeze(1)
                labels_at_level = gt_cls[:, point_start:point_end]
                offsets_at_level = gt_offsets[:, point_start:point_end]
                present_classes = labels_at_level.any(dim=0).any(dim=0)
                for class_id in torch.where(present_classes)[0].tolist():
                    selected = valid & labels_at_level[..., class_id].bool()
                    if selected.any():
                        values.setdefault((level, class_id), []).append(
                            torch.log1p(offsets_at_level[selected]).cpu()
                        )
                point_start = point_end
    return {
        "{}:{}".format(level, class_id): torch.cat(parts)
        for (level, class_id), parts in values.items()
    }


def make_bounds(values, quantile):
    return {
        key: torch.quantile(value, torch.tensor([quantile, 1.0 - quantile]), dim=0)
        for key, value in values.items()
    }


def collect_error_records(
        model, dataset, annotations, cfg, domain, split, levels, device):
    records = {key: [] for key in ("level", "class", "value", "error")}
    alpha = float(cfg["model"].get("pl_threshold", 0.7))
    batch_size = int(cfg.get("loader", {}).get("batch_size", 1))
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            raw_batch = [
                dataset[index]
                for index in range(start, min(start + batch_size, len(dataset)))
            ]
            batch = [move_video(video, device) for video in raw_batch]
            inputs, masks = model.preprocessing(
                batch, mode="train" if split == "train" else "validation",
            )
            logits, offsets, fpn_masks = model_outputs(model, inputs, masks, domain)
            points = model.point_generator(offsets)
            segments, labels = [], []
            for raw_video, video in zip(raw_batch, batch):
                segment, label = crop_targets(
                    annotations[raw_video["metadata"]["video_id"]],
                    video["metadata"], video["feats"].shape[-1], cfg, device,
                )
                segments.append(segment)
                labels.append(label)
            gt_cls, gt_offsets = model.label_points(points, segments, labels)
            gt_cls, gt_offsets = torch.stack(gt_cls), torch.stack(gt_offsets)
            point_start = 0
            for level, point in enumerate(points):
                point_end = point_start + point.shape[0]
                if level not in levels:
                    point_start = point_end
                    continue
                probabilities = logits[level].permute(0, 2, 1).sigmoid()
                confidence, predicted_class = probabilities.max(dim=-1)
                selected = fpn_masks[level].squeeze(1) & (confidence >= alpha)
                if domain == "source":
                    label_at_level = gt_cls[:, point_start:point_end]
                    selected &= label_at_level.gather(
                        -1, predicted_class.unsqueeze(-1),
                    ).squeeze(-1).bool()
                if selected.any():
                    predictions = torch.log1p(offsets[level].permute(0, 2, 1))
                    records["level"].append(torch.full(
                        (int(selected.sum()),), level, dtype=torch.long,
                    ))
                    records["class"].append(predicted_class[selected].cpu())
                    records["value"].append(predictions[selected].cpu())
                    records["error"].append(
                        (offsets[level].permute(0, 2, 1)
                         - gt_offsets[:, point_start:point_end])
                        .square().mean(dim=-1)[selected].cpu()
                    )
                point_start = point_end
    if not records["error"]:
        raise RuntimeError("No high-confidence anchors")
    return {key: torch.cat(value) for key, value in records.items()}


def range_mask(records, bounds):
    outlier = torch.zeros(records["error"].shape[0], dtype=torch.bool)
    for key, boundary in bounds.items():
        level, class_id = (int(value) for value in key.split(":"))
        selected = (records["level"] == level) & (records["class"] == class_id)
        if selected.any():
            values = records["value"][selected]
            outlier[selected] = ((values < boundary[0]) | (values > boundary[1])).any(dim=-1)
    return outlier


def candidate_metrics(records, bounds):
    outlier = range_mask(records, bounds)
    inlier = ~outlier
    return {
        "anchors": int(outlier.numel()),
        "outlier_fraction": float(outlier.float().mean()),
        "outlier_offset_mse": float(records["error"][outlier].mean()) if outlier.any() else None,
        "inlier_offset_mse": float(records["error"][inlier].mean()) if inlier.any() else None,
        "error_ratio": float(
            records["error"][outlier].mean() / records["error"][inlier].mean().clamp_min(1e-8)
        ) if outlier.any() and inlier.any() else None,
    }


def source_select(args):
    seed_everything(args.seed)
    cfg, checkpoint, model, annotations = load_model(args)
    levels = [int(value) for value in args.levels.split(",")]
    source_train = dataset_for(cfg, "source", "train")
    source_val = dataset_for(cfg, "source", "val")
    source_gt = collect_source_gt(
        model, source_train, annotations, cfg, levels, args.device,
    )
    val_records = collect_error_records(
        model, source_val, annotations, cfg, "source", "val", levels, args.device,
    )
    candidates, selected = [], None
    for quantile in parse_numbers(args.quantiles):
        bounds = make_bounds(source_gt, quantile)
        metrics = candidate_metrics(val_records, bounds)
        candidate = {"quantile": quantile, "metrics": metrics}
        candidates.append(candidate)
        if (
            metrics["error_ratio"] is not None
            and 0.01 <= metrics["outlier_fraction"] <= 0.5
            and (selected is None or metrics["error_ratio"] > selected["metrics"]["error_ratio"])
        ):
            selected = candidate
    if selected is None:
        raise RuntimeError("No source-valid feasible-range candidate")
    artifact = {
        "quantile": selected["quantile"], "levels": levels,
        "bounds": make_bounds(source_gt, selected["quantile"]),
        "config": args.config, "checkpoint": args.checkpoint,
    }
    torch.save(artifact, args.calibration_output)
    return {
        "audit": "source_gt_offset_feasible_range_selection",
        "checkpoint_epoch": checkpoint.get("epoch"),
        "source_gt_class_level_groups": len(source_gt),
        "source_val_anchors": int(val_records["error"].numel()),
        "candidates": candidates, "selected": selected,
        "selection_metric": "source_val_outlier_to_inlier_offset_mse_ratio",
        "calibration_output": args.calibration_output,
        "target_gt_usage": "none",
    }


def target_eval(args):
    artifact = torch.load(args.calibration, map_location="cpu")
    seed_everything(args.seed)
    cfg, checkpoint, model, annotations = load_model(args)
    target_val = dataset_for(cfg, "target", "val")
    records = collect_error_records(
        model, target_val, annotations, cfg, "target", "val", artifact["levels"], args.device,
    )
    return {
        "audit": "fixed_source_gt_offset_feasible_range_target_evaluation",
        "checkpoint_epoch": checkpoint.get("epoch"),
        "fixed_source_selection": {
            "quantile": artifact["quantile"], "levels": artifact["levels"],
        },
        "target_metrics": candidate_metrics(records, artifact["bounds"]),
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
    parser.add_argument("--quantiles", default="0.005,0.01,0.02,0.05")
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

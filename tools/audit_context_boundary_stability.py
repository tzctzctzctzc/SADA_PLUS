"""Audit Stage 2 boundary reliability from adjacent real-video context."""

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
from libs.models.SADA.stage1_utils import (
    context_view_on_cpu,
    interpolate_probabilities,
)
from libs.models.SADA.stage2_utils import aligned_boundary_iou_from_offsets
from libs.utils.utils import _update_config, read_yaml
from tools.audit_original_sada_pseudolabels import crop_targets, move_video


def parse_numbers(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def current_logits(model, inputs, masks, domain):
    if domain == "source":
        backbone, neck, head = (
            model.backbone_source, model.neck_source, model.cls_head,
        )
    else:
        backbone, neck, head = (
            model.backbone_target, model.neck_target, model.cls_head_target,
        )
    features, feature_masks, _ = backbone(inputs, masks)
    fpn_features, fpn_masks = neck(features, feature_masks)
    logits, _ = head(fpn_features, fpn_masks)
    return torch.cat([item.permute(0, 2, 1) for item in logits], dim=1)


def weighted_mse(error, weight):
    return float((weight * error).sum() / weight.sum().clamp_min(1e-6))


def build_context_views(raw_batch, batch, rho, generator):
    views, directions, offsets, valid = [], [], [], []
    for raw, video in zip(raw_batch, batch):
        original = raw["feats"]
        full = raw["full_feats"]
        start = int(video["metadata"].get("crop_start", 0))
        length = original.shape[-1]
        if not torch.equal(original, full[:, start:start + length]):
            raise RuntimeError("Context crop does not match full feature tensor")
        offset = max(1, int(round(rho * length)))
        legal = []
        if start - offset >= 0:
            legal.append(-1)
        if start + length + offset <= full.shape[-1]:
            legal.append(1)
        if legal:
            direction = legal[torch.randint(
                0, len(legal), (1,), generator=generator,
            ).item()]
            context_start = start + direction * offset
            is_valid = True
        else:
            direction = 0
            context_start = start
            is_valid = False
        context = dict(video)
        context["feats"] = context_view_on_cpu(
            original, full, context_start, length,
        )
        views.append(context)
        directions.append(direction)
        offsets.append(offset)
        valid.append(is_valid)
    return views, directions, offsets, valid


def collect_records(model, dataset, annotations, cfg, domain, split, levels, rho, seed, device):
    records = {key: [] for key in ("confidence", "stability", "error")}
    alpha = float(cfg["model"].get("pl_threshold", 0.7))
    batch_size = int(cfg.get("loader", {}).get("batch_size", 1))
    mode = "train" if split == "train" else "validation"
    teacher_backbone, teacher_neck, teacher_head = (
        model.backbone_pl, model.neck_pl, model.reg_head_pl,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            raw_batch = [
                dataset[index]
                for index in range(start, min(start + batch_size, len(dataset)))
            ]
            batch = [move_video(video, device) for video in raw_batch]
            context_batch, directions, offsets, context_valid = build_context_views(
                raw_batch, batch, rho, generator,
            )
            inputs0, masks0 = model.preprocessing(batch, mode=mode)
            inputs1, masks1 = model.preprocessing(context_batch, mode=mode)
            logits = current_logits(model, inputs0, masks0, domain)
            strict0 = model._stage1_strict_masks(masks0)
            strict1 = model._stage1_strict_masks(masks1)
            features0, teacher_masks0, _ = teacher_backbone(inputs0, masks0)
            fpn0, fpn_masks0 = teacher_neck(features0, teacher_masks0)
            offsets0, _ = teacher_head(fpn0, fpn_masks0)
            features1, teacher_masks1, _ = teacher_backbone(inputs1, masks1)
            fpn1, fpn_masks1 = teacher_neck(features1, teacher_masks1)
            offsets1, _ = teacher_head(fpn1, fpn_masks1)
            points = model.point_generator(fpn0)
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
            confidence, predicted_class = logits.sigmoid().max(dim=-1)
            offset = 0
            for level, point in enumerate(points):
                next_offset = offset + point.shape[0]
                if level not in levels:
                    offset = next_offset
                    continue
                aligned_rows, valid_rows = [], []
                for index, (direction, shift) in enumerate(zip(directions, offsets)):
                    aligned, aligned_valid = interpolate_probabilities(
                        offsets1[level][index:index + 1].permute(0, 2, 1),
                        strict1[level][index:index + 1].squeeze(1), point,
                        point[:, 0] - direction * shift,
                    )
                    aligned_rows.append(aligned)
                    valid_rows.append(aligned_valid)
                aligned = torch.cat(aligned_rows, dim=0)
                aligned_valid = torch.cat(valid_rows, dim=0)
                context_valid_mask = torch.tensor(
                    context_valid, device=device, dtype=torch.bool,
                )[:, None]
                valid = strict0[level].squeeze(1) & aligned_valid & context_valid_mask
                direct = offsets0[level].permute(0, 2, 1)
                stability = aligned_boundary_iou_from_offsets(direct, aligned)
                teacher_error = (
                    direct - gt_offsets[:, offset:next_offset]
                ).square().mean(dim=-1)
                selected = valid & (confidence[:, offset:next_offset] >= alpha)
                if domain == "source":
                    positive = gt_cls[:, offset:next_offset].sum(dim=-1) > 0
                    class_correct = gt_cls[:, offset:next_offset].gather(
                        -1, predicted_class[:, offset:next_offset].unsqueeze(-1),
                    ).squeeze(-1) > 0
                    selected &= positive & class_correct
                if selected.any():
                    records["confidence"].append(
                        confidence[:, offset:next_offset][selected].cpu()
                    )
                    records["stability"].append(stability[selected].cpu())
                    records["error"].append(teacher_error[selected].cpu())
                offset = next_offset
    if not records["error"]:
        raise RuntimeError("No selected anchors")
    return {key: torch.cat(value) for key, value in records.items()}


def candidate_metrics(records, power, fraction):
    scores = records["stability"] * records["confidence"].pow(power)
    keep = max(1, int(torch.ceil(torch.tensor(fraction * scores.numel())).item()))
    chosen = torch.argsort(scores, descending=True)[:keep]
    gate = torch.zeros_like(scores)
    gate[chosen] = 1.0
    weights = scores * gate
    return {
        "weighted_offset_mse": weighted_mse(records["error"], weights),
        "unweighted_offset_mse": float(records["error"].mean()),
        "kept_fraction": float(gate.mean()),
        "stability_mean": float(records["stability"].mean()),
        "kept_stability_mean": float(records["stability"][chosen].mean()),
    }


def source_select(args):
    seed_everything(args.seed)
    cfg = read_yaml(args.config)
    cfg["stage1"] = {"enabled": False, "perturbation": "shift"}
    cfg["dataset"]["stage1_context"] = True
    _update_config(cfg)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model = make_meta_arch(cfg["model_name"], cfg, device=args.device).to(args.device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    annotations = json.loads(Path(cfg["dataset"]["json_file"]).read_text(encoding="utf-8"))["database"]
    levels = [int(value) for value in args.levels.split(",")]
    train = make_dataset(cfg["dataset_name"], True, cfg["src_train_split"], **cfg["dataset"])
    val = make_dataset(cfg["dataset_name"], False, cfg["src_val_split"], **cfg["dataset"])
    train_records = collect_records(model, train, annotations, cfg, "source", "train", levels, args.rho, args.seed, args.device)
    val_records = collect_records(model, val, annotations, cfg, "source", "val", levels, args.rho, args.seed, args.device)
    candidates, best = [], None
    for power in parse_numbers(args.confidence_powers):
        for fraction in parse_numbers(args.keep_fractions):
            metrics = candidate_metrics(val_records, power, fraction)
            candidate = {"power": power, "fraction": fraction, "metrics": metrics}
            candidates.append(candidate)
            if best is None or metrics["weighted_offset_mse"] < best["metrics"]["weighted_offset_mse"]:
                best = candidate
    artifact = {
        "rho": args.rho, "seed": args.seed, "power": best["power"],
        "fraction": best["fraction"], "levels": levels,
        "config": args.config, "checkpoint": args.checkpoint,
    }
    torch.save(artifact, args.calibration_output)
    return {
        "audit": "source_only_context_boundary_stability_selection",
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
    seed_everything(artifact["seed"])
    cfg = read_yaml(args.config)
    cfg["stage1"] = {"enabled": False, "perturbation": "shift"}
    cfg["dataset"]["stage1_context"] = True
    _update_config(cfg)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model = make_meta_arch(cfg["model_name"], cfg, device=args.device).to(args.device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    annotations = json.loads(Path(cfg["dataset"]["json_file"]).read_text(encoding="utf-8"))["database"]
    val = make_dataset(cfg["dataset_name"], False, cfg["tgt_val_split"], **cfg["dataset"])
    records = collect_records(model, val, annotations, cfg, "target", "val", artifact["levels"], artifact["rho"], artifact["seed"], args.device)
    return {
        "audit": "fixed_source_only_context_boundary_stability_target_evaluation",
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
    parser.add_argument("--rho", type=float, default=64.0 / 2304.0)
    parser.add_argument("--confidence-powers", default="0,0.5,1")
    parser.add_argument("--keep-fractions", default="1,0.75,0.5")
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args()
    if args.mode == "source_select" and not args.calibration_output:
        parser.error("--calibration-output is required for source_select")
    if args.mode == "target_eval" and not args.calibration:
        parser.error("--calibration is required for target_eval")
    if not 0 < args.rho < 1:
        parser.error("--rho must be in (0, 1)")
    result = source_select(args) if args.mode == "source_select" else target_eval(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

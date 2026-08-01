"""Audit the historical Stage 2 proposal Oracle equivalence path."""

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import libs.models.SADA.meta_archs as sada_meta_archs
from libs.datasets import make_dataset
from libs.models.SADA import make_meta_arch
from libs.utils.utils import _update_config, read_yaml
from tools.audit_original_sada_pseudolabels import crop_targets, move_video


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def scalar_losses(losses):
    result = {}
    for key, value in losses.items():
        if torch.is_tensor(value) and value.numel() == 1:
            result[key] = float(value.detach().cpu())
    return result


def add_loss_sums(sums, losses):
    for key, value in losses.items():
        sums[key] = sums.get(key, 0.0) + value


def max_loss_diff(lhs, rhs, ignore):
    diffs = {}
    for key in sorted(set(lhs) & set(rhs)):
        if key in ignore:
            continue
        diffs[key] = abs(lhs[key] - rhs[key])
    return diffs


def grad_summary(model):
    tensors = 0
    finite_tensors = 0
    max_abs = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        tensors += 1
        grad = param.grad.detach()
        if torch.isfinite(grad).all():
            finite_tensors += 1
        if grad.numel():
            max_abs = max(max_abs, float(grad.abs().max().cpu()))
    return {
        "grad_tensors": tensors,
        "finite_grad_tensors": finite_tensors,
        "all_finite": tensors == finite_tensors,
        "max_abs_grad": max_abs,
    }


def configure_teacher(base_cfg, levels):
    cfg = copy.deepcopy(base_cfg)
    cfg["stage1"] = {"enabled": False, "perturbation": "shift"}
    cfg["stage2"] = {
        "enabled": True,
        "start_epoch": 5,
        "weight": 1.0,
        "levels": levels,
    }
    cfg["diagnostic"] = {"oracle_target_regression": "off"}
    _update_config(cfg)
    return cfg


def configure_student(base_cfg, levels, mode):
    cfg = copy.deepcopy(base_cfg)
    cfg["stage1"] = {"enabled": False, "perturbation": "shift"}
    cfg["stage2"] = {
        "enabled": False,
        "start_epoch": 5,
        "weight": 1.0,
        "levels": levels,
    }
    diagnostic = dict(cfg.get("diagnostic", {}))
    diagnostic.update({
        "oracle_routing": "off",
        "oracle_target_regression": mode,
        "oracle_target_regression_weight": 1.0,
        "target_regression_levels": levels,
    })
    cfg["diagnostic"] = diagnostic
    cfg.setdefault("train_cfg", {})["ema_active"] = True
    _update_config(cfg)
    return cfg


def load_models(base_cfg, checkpoint_path, device, levels):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    teacher = make_meta_arch(
        base_cfg["model_name"], configure_teacher(base_cfg, levels), device=device
    ).to(device)
    teacher.load_state_dict(checkpoint["ema_state_dict"])
    teacher.eval()

    students = {}
    for mode in ("clean_pseudo", "same_selection"):
        cfg = configure_student(base_cfg, levels, mode)
        model = make_meta_arch(cfg["model_name"], cfg, device=device).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        students[mode] = model
    return checkpoint, teacher, students


def rebuild_batch(raw_batch, annotations, cfg, device):
    batch = []
    for raw_video in raw_batch:
        video = move_video(raw_video, device)
        video_id = raw_video["metadata"]["video_id"]
        segments, labels = crop_targets(
            annotations[video_id],
            video["metadata"],
            video["feats"].shape[-1],
            cfg,
            device,
        )
        video["segments"] = segments
        video["labels"] = labels
        batch.append(video)
    return batch


class OracleCapture:
    def __init__(self):
        self.record = {}
        self._mask_fn = None
        self._target_fn = None

    def __enter__(self):
        self._mask_fn = sada_meta_archs.build_same_selection_oracle_mask
        self._target_fn = sada_meta_archs.select_same_selection_regression_targets

        def capture_mask(pseudo_labels, gt_labels, valid_mask, enabled_levels):
            mask = self._mask_fn(
                pseudo_labels, gt_labels, valid_mask, enabled_levels
            )
            self.record["mask"] = mask.detach().cpu()
            self.record["pseudo_positive"] = (
                pseudo_labels.sum(dim=-1) > 0
            ).detach().cpu()
            self.record["gt_positive"] = (
                gt_labels.sum(dim=-1) > 0
            ).detach().cpu()
            self.record["valid_mask"] = valid_mask.detach().cpu()
            self.record["enabled_levels"] = enabled_levels.detach().cpu()
            return mask

        def capture_targets(pseudo_offsets, gt_offsets, mode):
            target = self._target_fn(pseudo_offsets, gt_offsets, mode)
            self.record["pseudo_offsets"] = pseudo_offsets.detach().cpu()
            self.record["gt_offsets"] = gt_offsets.detach().cpu()
            self.record["selected_offsets"] = target.detach().cpu()
            return target

        sada_meta_archs.build_same_selection_oracle_mask = capture_mask
        sada_meta_archs.select_same_selection_regression_targets = capture_targets
        return self.record

    def __exit__(self, exc_type, exc, traceback):
        sada_meta_archs.build_same_selection_oracle_mask = self._mask_fn
        sada_meta_archs.select_same_selection_regression_targets = self._target_fn


def run_oracle_forward(model, source_batch, target_batch, pseudo_targets,
                       args, seed):
    model.zero_grad(set_to_none=True)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    with OracleCapture() as capture:
        output = model(
            source_batch,
            target_batch,
            mode=args.forward_mode,
            domain=args.domain,
            curr_epoch=args.curr_epoch,
            stage2_pseudo_targets=pseudo_targets,
        )
    losses = output[0] if isinstance(output, tuple) else output
    losses["downstream_loss"].backward()
    return {
        "losses": scalar_losses(losses),
        "capture": capture,
        "grad": grad_summary(model),
    }


def selected_offset_l1(capture):
    mask = capture["mask"]
    if not mask.any():
        return None
    pseudo_offsets = capture["pseudo_offsets"]
    gt_offsets = capture["gt_offsets"]
    return float((pseudo_offsets[mask] - gt_offsets[mask]).abs().mean())


def audit(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    levels = [int(value) for value in args.levels.split(",") if value]
    base_cfg = read_yaml(args.config)
    _update_config(base_cfg)
    annotations = json.loads(
        Path(base_cfg["dataset"]["json_file"]).read_text(encoding="utf-8")
    )["database"]

    source_dataset = make_dataset(
        base_cfg["dataset_name"], True, base_cfg["src_train_split"],
        **base_cfg["dataset"]
    )
    target_dataset = make_dataset(
        base_cfg["dataset_name"], True, base_cfg["tgt_train_split"],
        **base_cfg["dataset"]
    )
    batch_size = args.batch_size or int(base_cfg.get("loader", {}).get("batch_size", 1))
    checkpoint, teacher, students = load_models(
        base_cfg, args.checkpoint, args.device, levels
    )

    summary = {
        "batches": 0,
        "videos": 0,
        "pre_nms_proposals": 0,
        "post_nms_proposals": 0,
        "pseudo_positive_anchors": 0,
        "gt_positive_anchors": 0,
        "same_selection_anchors": 0,
        "mask_equal_batches": 0,
        "target_offset_l1_sum": 0.0,
        "target_offset_l1_batches": 0,
        "clean_near_zero_batches": 0,
        "backward_finite_batches": 0,
    }
    loss_sums = {"clean_pseudo": {}, "same_selection": {}}
    max_common_diffs = {}
    batch_records = []

    max_batches = args.max_batches or (
        (len(target_dataset) + batch_size - 1) // batch_size
    )

    for batch_index in range(max_batches):
        target_start = batch_index * batch_size
        if target_start >= len(target_dataset):
            break
        target_indices = list(range(
            target_start, min(target_start + batch_size, len(target_dataset))
        ))
        source_indices = [
            (target_start + offset) % len(source_dataset)
            for offset in range(len(target_indices))
        ]
        raw_source = [source_dataset[index] for index in source_indices]
        raw_target = [target_dataset[index] for index in target_indices]
        source_batch = rebuild_batch(raw_source, annotations, base_cfg, args.device)
        target_batch = rebuild_batch(raw_target, annotations, base_cfg, args.device)

        with torch.no_grad():
            post_nms, pre_nms = teacher.generate_stage2_pseudo_targets(
                target_batch, return_pre_nms=True
            )

        results = {}
        for mode, model in students.items():
            results[mode] = run_oracle_forward(
                model,
                source_batch,
                target_batch,
                post_nms,
                args,
                args.seed + batch_index,
            )

        clean = results["clean_pseudo"]
        same = results["same_selection"]
        clean_mask = clean["capture"]["mask"]
        same_mask = same["capture"]["mask"]
        masks_equal = bool(torch.equal(clean_mask, same_mask))
        if masks_equal:
            summary["mask_equal_batches"] += 1

        common_diffs = max_loss_diff(
            clean["losses"],
            same["losses"],
            ignore={"oracle_target_reg_loss", "downstream_loss"},
        )
        for key, value in common_diffs.items():
            max_common_diffs[key] = max(max_common_diffs.get(key, 0.0), value)

        for mode in ("clean_pseudo", "same_selection"):
            add_loss_sums(loss_sums[mode], results[mode]["losses"])

        l1 = selected_offset_l1(clean["capture"])
        if l1 is not None:
            summary["target_offset_l1_sum"] += l1
            summary["target_offset_l1_batches"] += 1

        clean_loss = clean["losses"].get("oracle_target_reg_loss", 0.0)
        if abs(clean_loss) <= args.near_zero_eps:
            summary["clean_near_zero_batches"] += 1
        if clean["grad"]["all_finite"] and same["grad"]["all_finite"]:
            summary["backward_finite_batches"] += 1

        pseudo_positive = clean["capture"]["pseudo_positive"]
        gt_positive = clean["capture"]["gt_positive"]
        summary["batches"] += 1
        summary["videos"] += len(target_indices)
        summary["pre_nms_proposals"] += sum(
            int(item["segments"].shape[0]) for item in pre_nms
        )
        summary["post_nms_proposals"] += sum(
            int(item["segments"].shape[0]) for item in post_nms
        )
        summary["pseudo_positive_anchors"] += int(pseudo_positive.sum())
        summary["gt_positive_anchors"] += int(gt_positive.sum())
        summary["same_selection_anchors"] += int(clean_mask.sum())

        batch_records.append({
            "batch_index": batch_index,
            "source_indices": source_indices,
            "target_indices": target_indices,
            "target_video_ids": [
                raw_target[offset]["metadata"]["video_id"]
                for offset in range(len(raw_target))
            ],
            "pre_nms_proposals": [
                int(item["segments"].shape[0]) for item in pre_nms
            ],
            "post_nms_proposals": [
                int(item["segments"].shape[0]) for item in post_nms
            ],
            "same_selection_anchors": int(clean_mask.sum()),
            "masks_equal": masks_equal,
            "target_offset_l1": l1,
            "losses": {
                "clean_pseudo": clean["losses"],
                "same_selection": same["losses"],
            },
            "common_loss_abs_diff": common_diffs,
            "grad": {
                "clean_pseudo": clean["grad"],
                "same_selection": same["grad"],
            },
        })

    loss_means = {}
    for mode, sums in loss_sums.items():
        loss_means[mode] = {
            key: value / max(summary["batches"], 1)
            for key, value in sums.items()
        }

    output = {
        "audit": "stage2_oracle_equivalence",
        "method": "EMA teacher -> post-NMS proposals -> label_points -> clean_pseudo/same_selection",
        "config": args.config,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "levels": levels,
        "batch_size": batch_size,
        "forward_mode": args.forward_mode,
        "domain": args.domain,
        "summary": {
            **summary,
            "all_masks_equal": summary["mask_equal_batches"] == summary["batches"],
            "all_backward_finite": (
                summary["backward_finite_batches"] == summary["batches"]
            ),
            "mask_equal_rate": ratio(
                summary["mask_equal_batches"], summary["batches"]
            ),
            "pseudo_anchor_foreground_precision": ratio(
                summary["same_selection_anchors"],
                summary["pseudo_positive_anchors"],
            ),
            "target_offset_l1_mean": ratio(
                summary["target_offset_l1_sum"],
                summary["target_offset_l1_batches"],
            ),
            "clean_near_zero_rate": ratio(
                summary["clean_near_zero_batches"], summary["batches"]
            ),
            "proposals_per_batch": ratio(
                summary["post_nms_proposals"], summary["batches"]
            ),
        },
        "loss_means": loss_means,
        "max_common_loss_abs_diff": max_common_diffs,
        "batches": batch_records,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=2)
    parser.add_argument("--levels", default="0,1,2")
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--curr-epoch", type=int, default=5)
    parser.add_argument("--forward-mode", default="validation")
    parser.add_argument("--domain", choices=("src", "tgt"), default="src")
    parser.add_argument("--near-zero-eps", type=float, default=1e-10)
    args = parser.parse_args()
    output = audit(args)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

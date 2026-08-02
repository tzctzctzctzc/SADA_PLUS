"""Audit why original SADA routes GT actions to background.

Target labels are used only after all frozen model predictions are produced.
They never select a checkpoint, threshold, candidate class, or training sample.
"""

import argparse
import collections
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

def best_positive_rank(probabilities, gt_labels):
    """Return the best rank and score among all positive classes per anchor."""
    if probabilities.shape != gt_labels.shape:
        raise ValueError("probabilities and gt_labels must have the same shape")
    if not torch.all(gt_labels.sum(dim=-1) > 0):
        raise ValueError("best_positive_rank requires at least one positive class")
    positive_scores = probabilities.masked_fill(~gt_labels.bool(), -1.0)
    best_score = positive_scores.max(dim=-1).values
    rank = (probabilities > best_score.unsqueeze(-1)).sum(dim=-1) + 1
    return rank, best_score


def majority_candidate(probability_views):
    """Choose a GT-independent top-1 majority, breaking ties by mean score."""
    if probability_views.ndim != 3:
        raise ValueError("probability_views must have shape [views, anchors, classes]")
    top1 = probability_views.argmax(dim=-1)
    class_support = F.one_hot(
        top1, num_classes=probability_views.shape[-1]
    ).sum(dim=0)
    max_support = class_support.max(dim=-1).values
    mean_probability = probability_views.mean(dim=0)
    tied_score = mean_probability.masked_fill(
        class_support != max_support.unsqueeze(-1), -1.0
    )
    candidate = tied_score.argmax(dim=-1)
    gather_index = candidate.view(1, -1, 1).expand(
        probability_views.shape[0], -1, 1
    )
    candidate_scores = probability_views.gather(-1, gather_index).squeeze(-1)
    return candidate, max_support, candidate_scores


def quantiles(values):
    if values.numel() == 0:
        return None
    values = values.float()
    qs = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=values.device)
    result = torch.quantile(values, qs).tolist()
    return {
        "mean": float(values.mean().item()),
        "q10": result[0],
        "q25": result[1],
        "q50": result[2],
        "q75": result[3],
        "q90": result[4],
    }


def ratio(numerator, denominator):
    return float(numerator) / denominator if denominator else None


def rank_summary(probabilities, gt_labels, mask, alpha):
    count = int(mask.sum().item())
    if not count:
        return {"n": 0}
    selected_prob = probabilities[mask]
    selected_gt = gt_labels[mask]
    rank, gt_score = best_positive_rank(selected_prob, selected_gt)
    max_score = selected_prob.max(dim=-1).values
    rank_counts = {
        "rank_1": int((rank == 1).sum().item()),
        "rank_2": int((rank == 2).sum().item()),
        "rank_3": int((rank == 3).sum().item()),
        "rank_gt_3": int((rank > 3).sum().item()),
    }
    return {
        "n": count,
        "counts": rank_counts,
        "rates": {key: value / count for key, value in rank_counts.items()},
        "top_1_recall": ratio(int((rank <= 1).sum().item()), count),
        "top_2_recall": ratio(int((rank <= 2).sum().item()), count),
        "top_3_recall": ratio(int((rank <= 3).sum().item()), count),
        "base_max_score_gap_to_alpha": quantiles(alpha - max_score),
        "best_gt_score_gap_to_alpha": quantiles(alpha - gt_score),
    }


def candidate_correct(candidate, gt_labels):
    return gt_labels.gather(-1, candidate.unsqueeze(-1)).squeeze(-1).bool()


def rule_summary(
    selected,
    correct,
    low_action,
    low_background,
    baseline_correct,
    all_gt_action_count,
):
    selected_action = selected & low_action
    selected_background = selected & low_background
    recovered_correct = int((selected_action & correct).sum().item())
    selected_total = int((selected_action | selected_background).sum().item())
    background_count = int(low_background.sum().item())
    low_action_count = int(low_action.sum().item())
    return {
        "selected_low_action": int(selected_action.sum().item()),
        "selected_low_background": int(selected_background.sum().item()),
        "recovered_correct": recovered_correct,
        "recovery_precision": ratio(recovered_correct, selected_total),
        "low_action_correct_recall": ratio(recovered_correct, low_action_count),
        "background_false_recovery_rate": ratio(
            int(selected_background.sum().item()), background_count
        ),
        "correct_da_utilization": ratio(
            baseline_correct + recovered_correct, all_gt_action_count
        ),
    }


def consensus_summary(
    probability_views,
    gt_labels,
    low_action,
    low_background,
    alpha,
    baseline_correct,
    all_gt_action_count,
):
    candidate, support, candidate_scores = majority_candidate(probability_views)
    correct = candidate_correct(candidate, gt_labels)
    mean_score = candidate_scores.mean(dim=0)
    any_high = candidate_scores.max(dim=0).values >= alpha
    mean_high = mean_score >= alpha
    output = {
        "views": int(probability_views.shape[0]),
        "candidate_score": quantiles(mean_score[low_action | low_background]),
        "rules": {},
    }
    for minimum_support in range(2, probability_views.shape[0] + 1):
        support_mask = support >= minimum_support
        key = "support_at_least_{}".format(minimum_support)
        output["rules"][key] = {
            "support_only": rule_summary(
                support_mask,
                correct,
                low_action,
                low_background,
                baseline_correct,
                all_gt_action_count,
            ),
            "support_and_any_view_high": rule_summary(
                support_mask & any_high,
                correct,
                low_action,
                low_background,
                baseline_correct,
                all_gt_action_count,
            ),
            "support_and_mean_high": rule_summary(
                support_mask & mean_high,
                correct,
                low_action,
                low_background,
                baseline_correct,
                all_gt_action_count,
            ),
        }
    return output


def load_models(cfg, view_specs, device):
    from libs.models.SADA import make_meta_arch

    by_checkpoint = collections.defaultdict(list)
    for name, checkpoint_path, state_key in view_specs:
        by_checkpoint[checkpoint_path].append((name, state_key))

    models = {}
    metadata = {}
    for checkpoint_path, checkpoint_views in by_checkpoint.items():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        for name, state_key in checkpoint_views:
            if state_key not in checkpoint:
                raise KeyError("{} has no {}".format(checkpoint_path, state_key))
            model = make_meta_arch(cfg["model_name"], cfg, device=device).to(device)
            checkpoint_state = checkpoint[state_key]
            model_state = model.state_dict()
            compatible_state = {
                key: value
                for key, value in checkpoint_state.items()
                if key in model_state and model_state[key].shape == value.shape
            }
            required_prefixes = (
                "backbone_target.",
                "neck_target.",
                "cls_head_target.",
            )
            for prefix in required_prefixes:
                required_keys = [
                    key for key in checkpoint_state if key.startswith(prefix)
                ]
                if not required_keys or any(
                    key not in compatible_state for key in required_keys
                ):
                    raise RuntimeError(
                        "incompatible checkpoint weights for {}".format(prefix)
                    )
            model.load_state_dict(compatible_state, strict=False)
            model.eval()
            models[name] = model
            metadata[name] = {
                "checkpoint": checkpoint_path,
                "checkpoint_epoch": checkpoint.get("epoch"),
                "state_key": state_key,
                "loaded_tensor_count": len(compatible_state),
                "skipped_incompatible_tensor_count": len(checkpoint_state) - len(compatible_state),
            }
        del checkpoint
    return models, metadata


def forward_probabilities(model, inputs, masks):
    backbone_feats, backbone_masks, _ = model.backbone_target(inputs, masks)
    fpn_feats, fpn_masks = model.neck_target(backbone_feats, backbone_masks)
    logits, _ = model.cls_head_target(fpn_feats, fpn_masks)
    probabilities = [value.permute(0, 2, 1).sigmoid() for value in logits]
    return probabilities, fpn_feats, fpn_masks


def parse_views(raw_views):
    views = []
    seen = set()
    for name, checkpoint, state_key in raw_views:
        if name in seen:
            raise ValueError("duplicate view name: {}".format(name))
        if state_key not in ("state_dict", "ema_state_dict"):
            raise ValueError("state key must be state_dict or ema_state_dict")
        seen.add(name)
        views.append((name, checkpoint, state_key))
    return views


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--view",
        action="append",
        nargs=3,
        metavar=("NAME", "CHECKPOINT", "STATE_KEY"),
        required=True,
    )
    parser.add_argument("--base-view", default="e10_student")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max-videos", type=int, default=None)
    args = parser.parse_args()

    from libs.datasets import make_dataset
    from libs.utils.utils import _update_config, read_yaml
    from tools.audit_original_sada_pseudolabels import crop_targets, move_video

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    view_specs = parse_views(args.view)
    if args.base_view not in {item[0] for item in view_specs}:
        raise ValueError("base view is missing from --view")

    cfg = read_yaml(args.config)
    cfg["stage1"] = {"enabled": False, "perturbation": "shift"}
    _update_config(cfg)
    dataset = make_dataset(
        cfg["dataset_name"], True, cfg["tgt_train_split"], **cfg["dataset"]
    )
    annotations = json.load(
        open(cfg["dataset"]["json_file"], encoding="utf-8")
    )["database"]
    batch_size = args.batch_size or int(cfg.get("loader", {}).get("batch_size", 1))
    video_count = min(len(dataset), args.max_videos or len(dataset))
    alpha = float(cfg["model"].get("pl_threshold", 0.7))
    device = args.device

    models, view_metadata = load_models(cfg, view_specs, device)
    base_model = models[args.base_view]
    probability_records = {name: [] for name in models}
    gt_records = []
    level_records = []

    with torch.inference_mode():
        for batch_start in range(0, video_count, batch_size):
            raw_videos = [
                dataset[index]
                for index in range(
                    batch_start, min(batch_start + batch_size, video_count)
                )
            ]
            videos = [move_video(video, device) for video in raw_videos]
            inputs, masks = base_model.preprocessing(videos, mode="validation")

            batch_probabilities = {}
            base_fpn_feats = None
            base_fpn_masks = None
            forward_names = [args.base_view] + [
                name for name in models if name != args.base_view
            ]
            for name in forward_names:
                model = models[name]
                probabilities, fpn_feats, fpn_masks = forward_probabilities(
                    model, inputs, masks
                )
                batch_probabilities[name] = probabilities
                if name == args.base_view:
                    base_fpn_feats = fpn_feats
                    base_fpn_masks = fpn_masks
                else:
                    for level in args.levels:
                        if not torch.equal(fpn_masks[level], base_fpn_masks[level]):
                            raise RuntimeError("view FPN masks differ at level {}".format(level))

            points = base_model.point_generator(base_fpn_feats)
            gt_segments, gt_labels = [], []
            for raw_video, video in zip(raw_videos, videos):
                video_id = video["metadata"]["video_id"]
                segments, labels = crop_targets(
                    annotations[video_id],
                    video["metadata"],
                    video["feats"].shape[-1],
                    cfg,
                    device,
                )
                if not torch.equal(segments, raw_video["segments"].to(device)) or not torch.equal(
                    labels, raw_video["labels"].to(device)
                ):
                    raise RuntimeError("target GT reconstruction mismatch for {}".format(video_id))
                gt_segments.append(segments)
                gt_labels.append(labels)
            gt_cls, _ = base_model.label_points(points, gt_segments, gt_labels)
            gt_cls = torch.stack(gt_cls).bool()

            start = 0
            for level, point_grid in enumerate(points):
                length = point_grid.shape[0]
                if level in args.levels:
                    valid = base_fpn_masks[level].squeeze(1)
                    labels = gt_cls[:, start:start + length]
                    gt_records.append(labels[valid].cpu())
                    level_records.append(
                        torch.full(
                            (int(valid.sum().item()),), level, dtype=torch.long
                        )
                    )
                    for name in models:
                        probability_records[name].append(
                            batch_probabilities[name][level][valid].cpu()
                        )
                start += length

    gt_labels = torch.cat(gt_records, dim=0)
    levels = torch.cat(level_records, dim=0)
    probabilities = {
        name: torch.cat(records, dim=0)
        for name, records in probability_records.items()
    }
    ordered_names = [name for name, _, _ in view_specs]
    all_views = torch.stack([probabilities[name] for name in ordered_names])
    base_probability = probabilities[args.base_view]
    base_score, base_class = base_probability.max(dim=-1)
    gt_action = gt_labels.sum(dim=-1) > 0
    gt_background = ~gt_action
    base_high = base_score >= alpha
    base_low = ~base_high
    base_correct = candidate_correct(base_class, gt_labels)
    low_action = base_low & gt_action
    low_background = base_low & gt_background
    baseline_correct = int((base_high & base_correct).sum().item())
    all_gt_action_count = int(gt_action.sum().item())

    analysis_a = {
        "global": rank_summary(
            base_probability, gt_labels, low_action, alpha
        ),
        "by_level": {
            str(level): rank_summary(
                base_probability,
                gt_labels,
                low_action & (levels == level),
                alpha,
            )
            for level in args.levels
        },
        "by_base_max_score_gap": {},
    }
    gap = alpha - base_score
    gap_bins = [
        ("le_0.02", gap <= 0.02),
        ("0.02_0.05", (gap > 0.02) & (gap <= 0.05)),
        ("0.05_0.10", (gap > 0.05) & (gap <= 0.10)),
        ("gt_0.10", gap > 0.10),
    ]
    for name, gap_mask in gap_bins:
        analysis_a["by_base_max_score_gap"][name] = rank_summary(
            base_probability, gt_labels, low_action & gap_mask, alpha
        )

    view_results = {}
    for name in ordered_names:
        view_probability = probabilities[name]
        view_score, view_class = view_probability.max(dim=-1)
        view_correct = candidate_correct(view_class, gt_labels)
        low_rank, _ = best_positive_rank(
            view_probability[low_action], gt_labels[low_action]
        )
        selected = view_score >= alpha
        selected_low = selected & (low_action | low_background)
        recovered_correct = int((selected & low_action & view_correct).sum().item())
        view_results[name] = {
            "low_action_top_1_recall": ratio(int((low_rank <= 1).sum().item()), int(low_action.sum().item())),
            "low_action_top_2_recall": ratio(int((low_rank <= 2).sum().item()), int(low_action.sum().item())),
            "low_action_top_3_recall": ratio(int((low_rank <= 3).sum().item()), int(low_action.sum().item())),
            "high_confidence_rule": {
                "selected_low_action": int((selected & low_action).sum().item()),
                "selected_low_background": int((selected & low_background).sum().item()),
                "recovered_correct": recovered_correct,
                "recovery_precision": ratio(recovered_correct, int(selected_low.sum().item())),
                "low_action_correct_recall": ratio(recovered_correct, int(low_action.sum().item())),
                "background_false_recovery_rate": ratio(
                    int((selected & low_background).sum().item()),
                    int(low_background.sum().item()),
                ),
                "correct_da_utilization": ratio(
                    baseline_correct + recovered_correct, all_gt_action_count
                ),
            },
        }

    full_candidate, _, _ = majority_candidate(all_views)
    full_correct = candidate_correct(full_candidate, gt_labels)
    gt_top1_support = torch.stack([
        candidate_correct(probabilities[name].argmax(dim=-1), gt_labels)
        for name in ordered_names
    ]).sum(dim=0)
    gt_top3_support = torch.stack([
        (
            gt_labels.gather(
                -1, probabilities[name].topk(3, dim=-1).indices
            ).sum(dim=-1) > 0
        )
        for name in ordered_names
    ]).sum(dim=0)

    student_names = [name for name in ordered_names if name.endswith("_student")]
    ema_names = [name for name in ordered_names if name.endswith("_ema")]
    consensus = {
        "all_views": consensus_summary(
            all_views,
            gt_labels,
            low_action,
            low_background,
            alpha,
            baseline_correct,
            all_gt_action_count,
        )
    }
    if len(student_names) >= 2:
        consensus["student_checkpoints"] = consensus_summary(
            torch.stack([probabilities[name] for name in student_names]),
            gt_labels,
            low_action,
            low_background,
            alpha,
            baseline_correct,
            all_gt_action_count,
        )
    if len(ema_names) >= 2:
        consensus["ema_checkpoints"] = consensus_summary(
            torch.stack([probabilities[name] for name in ema_names]),
            gt_labels,
            low_action,
            low_background,
            alpha,
            baseline_correct,
            all_gt_action_count,
        )
    epoch_pairs = {}
    for student_name in student_names:
        prefix = student_name[:-len("_student")]
        ema_name = prefix + "_ema"
        if ema_name in probabilities:
            epoch_pairs[prefix] = consensus_summary(
                torch.stack([probabilities[student_name], probabilities[ema_name]]),
                gt_labels,
                low_action,
                low_background,
                alpha,
                baseline_correct,
                all_gt_action_count,
            )

    base_rank, _ = best_positive_rank(
        base_probability[low_action], gt_labels[low_action]
    )
    support_histograms = {}
    for label, values in (
        ("gt_top1_view_support", gt_top1_support[low_action]),
        ("gt_top3_view_support", gt_top3_support[low_action]),
    ):
        support_histograms[label] = {
            str(value): int((values == value).sum().item())
            for value in range(len(ordered_names) + 1)
        }

    result = {
        "audit": "low_confidence_gt_action_topk_and_model_consensus",
        "seed": args.seed,
        "videos": video_count,
        "levels": args.levels,
        "alpha": alpha,
        "base_view": args.base_view,
        "views": view_metadata,
        "protocol": {
            "cohort": "base-view max score < alpha on ordinary valid target-train anchors",
            "gt_rank": "best rank among all positive GT classes",
            "model_mode": "eval; identical frozen crop and anchor coordinates for every view",
            "checkpoint_selection": "fixed e5/e10/e50 student and EMA; target GT is audit-only",
            "consensus_candidate": "top-1 mode; ties broken by mean candidate probability",
        },
        "cohort_counts": {
            "valid": int(gt_labels.shape[0]),
            "all_gt_action": all_gt_action_count,
            "all_gt_background": int(gt_background.sum().item()),
            "base_low_gt_action": int(low_action.sum().item()),
            "base_low_gt_background": int(low_background.sum().item()),
            "base_high_correct_action": baseline_correct,
            "base_correct_da_utilization": ratio(baseline_correct, all_gt_action_count),
            "base_gt_action_background_leakage": ratio(int(low_action.sum().item()), all_gt_action_count),
        },
        "analysis_a_base_topk": analysis_a,
        "oracle_topk_ceiling": {
            "warning": "Uses GT to choose a class inside top-k; not deployable.",
            "top_1_correct_da_utilization": ratio(
                baseline_correct + int((base_rank <= 1).sum().item()), all_gt_action_count
            ),
            "top_2_correct_da_utilization": ratio(
                baseline_correct + int((base_rank <= 2).sum().item()), all_gt_action_count
            ),
            "top_3_correct_da_utilization": ratio(
                baseline_correct + int((base_rank <= 3).sum().item()), all_gt_action_count
            ),
        },
        "analysis_b_views": view_results,
        "gt_support_across_views": support_histograms,
        "deployable_consensus": consensus,
        "student_ema_pairs": epoch_pairs,
        "all_view_majority_candidate_accuracy_on_low_action": ratio(
            int((full_correct & low_action).sum().item()), int(low_action.sum().item())
        ),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "cohort_counts": result["cohort_counts"],
        "base_topk": result["analysis_a_base_topk"]["global"],
        "oracle_topk_ceiling": result["oracle_topk_ceiling"],
        "majority_candidate_accuracy": result["all_view_majority_candidate_accuracy_on_low_action"],
    }, indent=2))
    print("wrote", args.output)


if __name__ == "__main__":
    main()

"""Train a source-only action pseudo-label reliability probe.

The frozen SADA checkpoint produces all features before any target labels are
read. Source GT trains the probe; target GT is used only for offline metrics.
"""

import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from libs.datasets import make_dataset
from libs.models.SADA import make_meta_arch
from libs.models.SADA.stage1_utils import (
    context_view_on_cpu,
    interpolate_probabilities,
)
from libs.utils.utils import _update_config, read_yaml
from tools.audit_original_sada_pseudolabels import crop_targets, move_video


RETENTION_FRACTIONS = (0.8, 0.6, 0.4, 0.2)
MODEL_FEATURES = {
    "confidence": ("score",),
    "static_probe": (
        "score",
        "margin",
        "entropy",
        "actionness",
        "prototype_cosine",
        "level_0",
        "level_1",
        "level_2",
        "level_3",
        "level_4",
        "level_5",
    ),
    "full_probe": (
        "score",
        "margin",
        "entropy",
        "actionness",
        "prototype_cosine",
        "level_0",
        "level_1",
        "level_2",
        "level_3",
        "level_4",
        "level_5",
        "context_consistent",
        "context_score_delta",
        "representation_cosine",
        "representation_mse",
    ),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--levels", default="0,1,2")
    parser.add_argument("--probe-max-iter", type=int, default=30)
    parser.add_argument(
        "--probe-l2-grid", default="0.001,0.01,0.1,1,10"
    )
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_context_batch(raw_videos, videos, rho, rho_max, generator):
    context_videos = []
    directions = []
    offsets = []
    valid = []
    for raw_video, video in zip(raw_videos, videos):
        original = raw_video["feats"]
        full = raw_video["full_feats"]
        start = int(video["metadata"].get("crop_start", 0))
        length = original.shape[-1]
        if not torch.equal(original, full[:, start:start + length]):
            raise RuntimeError("Context crop does not match full features")

        offset = max(1, int(round(rho * length)))
        legal = []
        if offset / float(length) <= rho_max:
            if start - offset >= 0:
                legal.append(-1)
            if start + length + offset <= full.shape[-1]:
                legal.append(1)
        if legal:
            pick = torch.randint(0, len(legal), (1,), generator=generator).item()
            direction = legal[pick]
            context_start = start + direction * offset
            is_valid = True
        else:
            direction = 0
            context_start = start
            is_valid = False

        context_video = dict(video)
        context_video["feats"] = context_view_on_cpu(
            original, full, context_start, length
        )
        context_videos.append(context_video)
        directions.append(direction)
        offsets.append(offset)
        valid.append(is_valid)
    return context_videos, directions, offsets, valid


def verify_targets(raw_videos, videos, annotations, cfg, device):
    gt_segments = []
    gt_labels = []
    for raw_video, video in zip(raw_videos, videos):
        video_id = video["metadata"]["video_id"]
        segments, labels = crop_targets(
            annotations[video_id],
            video["metadata"],
            video["feats"].shape[-1],
            cfg,
            device,
        )
        expected_segments = raw_video["segments"].to(device)
        expected_labels = raw_video["labels"].to(device)
        if (
            segments.shape != expected_segments.shape
            or not torch.equal(segments, expected_segments)
            or not torch.equal(labels, expected_labels)
        ):
            raise RuntimeError("GT reconstruction mismatch for {}".format(video_id))
        gt_segments.append(segments)
        gt_labels.append(labels)
    return gt_segments, gt_labels


def domain_modules(model, domain):
    if domain == "source":
        return model.backbone_source, model.neck_source, model.cls_head, model.reg_head
    return (
        model.backbone_target,
        model.neck_target,
        model.cls_head_target,
        model.reg_head_target,
    )


def forward_view(model, videos, modules):
    backbone, neck, cls_head, reg_head = modules
    inputs, masks = model.preprocessing(videos, mode="validation")
    strict = model._stage1_strict_masks(masks)
    feats, backbone_masks, _ = backbone(inputs, masks)
    fpn, fpn_masks = neck(feats, backbone_masks)
    logits, head_feats = cls_head(fpn, fpn_masks)
    _, actionness = reg_head(fpn, fpn_masks)
    return strict, fpn, fpn_masks, logits, head_feats, actionness


def empty_record_lists():
    return {
        key: []
        for key in (
            "score",
            "margin",
            "entropy",
            "actionness",
            "context_consistent",
            "context_score_delta",
            "representation_cosine",
            "representation_mse",
            "level",
            "pseudo_class",
            "foreground",
            "correct",
            "candidate",
            "feature",
        )
    }


def collect_records(
    model,
    dataset,
    annotations,
    cfg,
    domain,
    seed,
    batch_size,
    alpha,
    levels,
    collect_prototypes=False,
):
    random.seed(seed)
    torch.manual_seed(seed)
    modules = domain_modules(model, domain)
    device = next(model.parameters()).device
    rho = float(cfg["stage1"].get("context_rho", 64.0 / 2304.0))
    rho_max = float(cfg["stage1"].get("context_rho_max", 2.0 * rho))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    records = empty_record_lists()
    num_classes = int(model.num_class_orig)
    feature_dim = int(cfg["model"]["head_dim"])
    prototype_sum = torch.zeros(num_classes, feature_dim, device=device)
    prototype_count = torch.zeros(num_classes, device=device)
    videos_total = 0
    context_valid_videos = 0
    eligible_strict_n = 0
    common_valid_n = 0

    with torch.no_grad():
        for batch_start in range(0, len(dataset), batch_size):
            raw_videos = [
                dataset[index]
                for index in range(
                    batch_start, min(batch_start + batch_size, len(dataset))
                )
            ]
            videos = [move_video(video, str(device)) for video in raw_videos]
            context_videos, directions, offsets, context_valid = build_context_batch(
                raw_videos, videos, rho, rho_max, generator
            )
            context_valid_rows = torch.tensor(
                context_valid, dtype=torch.bool, device=device
            ).unsqueeze(1)
            videos_total += len(videos)
            context_valid_videos += sum(context_valid)

            strict0, fpn0, _, logits0, features0, actionness0 = forward_view(
                model, videos, modules
            )
            strict1, _, _, logits1, features1, _ = forward_view(
                model, context_videos, modules
            )
            points = model.point_generator(fpn0)
            gt_segments, gt_labels = verify_targets(
                raw_videos, videos, annotations, cfg, device
            )
            gt_cls, _ = model.label_points(points, gt_segments, gt_labels)
            gt_cls = torch.stack(gt_cls)

            label_start = 0
            for level, point_grid in enumerate(points):
                length = point_grid.shape[0]
                prob0 = logits0[level].permute(0, 2, 1).sigmoid()
                prob1 = logits1[level].permute(0, 2, 1).sigmoid()
                feature0 = features0[level].permute(0, 2, 1)
                feature1 = features1[level].permute(0, 2, 1)
                labels = gt_cls[:, label_start:label_start + length]
                strict_valid0 = strict0[level].squeeze(1)

                if level not in levels:
                    label_start += length
                    continue

                if collect_prototypes:
                    flat_valid = strict_valid0.reshape(-1)
                    flat_labels = labels.reshape(-1, num_classes)[flat_valid].float()
                    flat_features = feature0.reshape(-1, feature_dim)[flat_valid]
                    prototype_sum += flat_labels.transpose(0, 1).matmul(flat_features)
                    prototype_count += flat_labels.sum(dim=0)

                aligned_prob_rows = []
                aligned_feature_rows = []
                aligned_valid_rows = []
                for batch_index, (direction, offset) in enumerate(
                    zip(directions, offsets)
                ):
                    query = point_grid[:, 0] - direction * offset
                    aligned_prob, aligned_valid = interpolate_probabilities(
                        prob1[batch_index:batch_index + 1],
                        strict1[level][batch_index:batch_index + 1].squeeze(1),
                        point_grid,
                        query,
                    )
                    aligned_feature, _ = interpolate_probabilities(
                        feature1[batch_index:batch_index + 1],
                        strict1[level][batch_index:batch_index + 1].squeeze(1),
                        point_grid,
                        query,
                    )
                    aligned_prob_rows.append(aligned_prob)
                    aligned_feature_rows.append(aligned_feature)
                    aligned_valid_rows.append(aligned_valid)
                aligned_prob1 = torch.cat(aligned_prob_rows, dim=0)
                aligned_feature1 = torch.cat(aligned_feature_rows, dim=0)
                aligned_valid = torch.cat(aligned_valid_rows, dim=0)

                eligible = strict_valid0 & context_valid_rows
                common_valid = eligible & aligned_valid
                score, pseudo_class = prob0.max(dim=-1)
                top2 = prob0.topk(k=2, dim=-1).values
                margin = top2[..., 0] - top2[..., 1]
                entropy = -(
                    prob0.clamp(1e-7, 1 - 1e-7) * prob0.clamp(1e-7, 1 - 1e-7).log()
                    + (1 - prob0).clamp(1e-7, 1 - 1e-7)
                    * (1 - prob0).clamp(1e-7, 1 - 1e-7).log()
                ).mean(dim=-1)
                context_class = aligned_prob1.argmax(dim=-1)
                context_class_score = aligned_prob1.gather(
                    -1, pseudo_class.unsqueeze(-1)
                ).squeeze(-1)
                representation_cosine = 1.0 - F.cosine_similarity(
                    feature0, aligned_feature1, dim=-1, eps=1e-8
                )
                representation_mse = F.mse_loss(
                    feature0, aligned_feature1, reduction="none"
                ).mean(dim=-1)
                foreground = labels.sum(dim=-1) > 0
                correct = labels.gather(
                    -1, pseudo_class.unsqueeze(-1)
                ).squeeze(-1) > 0
                eligible_strict_n += int(eligible.sum().item())
                common_valid_n += int(common_valid.sum().item())
                values = {
                    "score": score,
                    "margin": margin,
                    "entropy": entropy,
                    "actionness": actionness0[level].squeeze(1),
                    "context_consistent": pseudo_class == context_class,
                    "context_score_delta": (score - context_class_score).abs(),
                    "representation_cosine": representation_cosine,
                    "representation_mse": representation_mse,
                    "level": torch.full_like(pseudo_class, level),
                    "pseudo_class": pseudo_class,
                    "foreground": foreground,
                    "correct": correct,
                    "candidate": score >= alpha,
                    "feature": feature0,
                }
                for key, value in values.items():
                    records[key].append(value[common_valid].detach().cpu())
                label_start += length

    if not records["score"]:
        raise RuntimeError("No records were collected")
    output = {key: torch.cat(value) for key, value in records.items()}
    stats = {
        "videos": videos_total,
        "context_valid_videos": context_valid_videos,
        "context_valid_video_ratio": context_valid_videos / max(videos_total, 1),
        "eligible_strict_n": eligible_strict_n,
        "common_valid_n": common_valid_n,
        "common_overlap_ratio": common_valid_n / max(eligible_strict_n, 1),
        "candidate_n": int(output["candidate"].sum().item()),
        "record_n": int(output["score"].numel()),
    }
    if collect_prototypes:
        return output, stats, prototype_sum.cpu(), prototype_count.cpu()
    return output, stats


def add_prototype_distance(records, prototype_sum, prototype_count):
    prototypes = prototype_sum / prototype_count.clamp_min(1).unsqueeze(1)
    selected = prototypes[records["pseudo_class"]]
    records["prototype_cosine"] = 1.0 - F.cosine_similarity(
        records["feature"], selected, dim=-1, eps=1e-8
    )
    records.pop("feature")


def feature_matrix(records, feature_names, n_levels):
    columns = []
    for name in feature_names:
        if name.startswith("level_"):
            level = int(name.split("_")[1])
            column = (records["level"] == level).float()
        else:
            column = records[name].float()
        columns.append(column)
    return torch.stack(columns, dim=1)


def select_records(records, selected):
    return {key: value[selected] for key, value in records.items()}


def train_logistic_probe(x, y, max_iter, l2):
    mean = x.mean(dim=0)
    std = x.std(dim=0, unbiased=False).clamp_min(1e-6)
    normalized = (x - mean) / std
    normalized = normalized.double()
    target = y.double()
    weight = torch.zeros(normalized.shape[1], dtype=torch.double, requires_grad=True)
    bias = torch.zeros((), dtype=torch.double, requires_grad=True)
    positives = target.sum().clamp_min(1.0)
    negatives = (1.0 - target).sum().clamp_min(1.0)
    pos_weight = negatives / positives
    optimizer = torch.optim.LBFGS(
        (weight, bias), max_iter=max_iter, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        logits = normalized.matmul(weight) + bias
        loss = F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=pos_weight
        ) + l2 * weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return {
        "mean": mean,
        "std": std,
        "weight": weight.detach().float(),
        "bias": float(bias.detach().item()),
        "positive_weight": float(pos_weight.item()),
    }


def apply_probe(x, probe):
    normalized = (x - probe["mean"]) / probe["std"]
    return normalized.matmul(probe["weight"]) + probe["bias"]


def ranking_metrics(scores, labels):
    labels = labels.bool()
    positives = int(labels.sum().item())
    negatives = int((~labels).sum().item())
    order = torch.argsort(scores, descending=True)
    sorted_labels = labels[order].float()
    cumulative_true = sorted_labels.cumsum(dim=0)
    precision = cumulative_true / torch.arange(
        1, len(labels) + 1, dtype=torch.float32
    )
    average_precision = float(
        (precision * sorted_labels).sum().item() / max(positives, 1)
    )
    if positives and negatives:
        false_positive = (1.0 - sorted_labels).cumsum(dim=0) / negatives
        true_positive = cumulative_true / positives
        false_positive = torch.cat((torch.zeros(1), false_positive))
        true_positive = torch.cat((torch.zeros(1), true_positive))
        auroc = float(torch.trapz(true_positive, false_positive).item())
    else:
        auroc = None
    return {"auroc": auroc, "average_precision": average_precision}


def subset_summary(indices, records):
    correct = records["correct"][indices].bool()
    foreground = records["foreground"][indices].bool()
    all_correct = int(records["correct"].sum().item())
    foreground_n = int(foreground.sum().item())
    return {
        "n": int(indices.numel()),
        "pseudo_correct_rate": float(correct.float().mean().item()),
        "correct_candidate_coverage": int(correct.sum().item()) / max(all_correct, 1),
        "foreground_rate": float(foreground.float().mean().item()),
        "class_accuracy_given_foreground": (
            float(correct[foreground].float().mean().item()) if foreground_n else None
        ),
    }


def evaluate_scores(scores, records):
    labels = records["correct"].bool()
    result = ranking_metrics(scores, labels)
    result["all_candidates"] = subset_summary(
        torch.arange(scores.numel()), records
    )
    result["retention"] = {}
    order = torch.argsort(scores, descending=True)
    for fraction in RETENTION_FRACTIONS:
        keep = max(1, int(math.ceil(fraction * scores.numel())))
        key = "retain_{:03d}".format(int(round(fraction * 100)))
        result["retention"][key] = subset_summary(order[:keep], records)
    return result


def evaluate_binary_scores(scores, labels):
    labels = labels.bool()
    result = ranking_metrics(scores, labels)
    result["all_positive_rate"] = float(labels.float().mean().item())
    result["retention"] = {}
    order = torch.argsort(scores, descending=True)
    positives = int(labels.sum().item())
    for fraction in RETENTION_FRACTIONS:
        keep = max(1, int(math.ceil(fraction * scores.numel())))
        chosen = labels[order[:keep]]
        key = "retain_{:03d}".format(int(round(fraction * 100)))
        result["retention"][key] = {
            "n": keep,
            "positive_rate": float(chosen.float().mean().item()),
            "positive_coverage": int(chosen.sum().item()) / max(positives, 1),
        }
    return result


def serialize_probe(probe, feature_names):
    return {
        "feature_names": list(feature_names),
        "mean": probe["mean"].tolist(),
        "std": probe["std"].tolist(),
        "weight": probe["weight"].tolist(),
        "bias": probe["bias"],
        "positive_weight": probe["positive_weight"],
    }


def select_probe_by_source_val(
    train_x, train_y, val_x, val_y, l2_values, max_iter
):
    candidates = []
    best = None
    for l2 in l2_values:
        probe = train_logistic_probe(train_x, train_y, max_iter, l2)
        val_scores = apply_probe(val_x, probe)
        val_metrics = ranking_metrics(val_scores, val_y)
        candidate = {
            "l2": l2,
            "source_val_average_precision": val_metrics["average_precision"],
            "source_val_auroc": val_metrics["auroc"],
        }
        candidates.append(candidate)
        ranking_key = (candidate["source_val_average_precision"], l2)
        if best is None or ranking_key > best[0]:
            best = (ranking_key, probe, l2)
    return best[1], best[2], candidates


def main():
    args = parse_args()
    seed_everything(args.seed)
    cfg = read_yaml(args.config)
    cfg["stage1"] = {
        "enabled": False,
        "perturbation": "context",
        "seed": args.seed,
    }
    cfg["dataset"]["stage1_context"] = True
    _update_config(cfg)
    batch_size = args.batch_size or int(cfg.get("loader", {}).get("batch_size", 1))
    alpha = float(cfg["model"].get("pl_threshold", 0.7))
    n_levels = len(cfg["model"]["regression_range"])
    levels = tuple(int(value) for value in args.levels.split(","))
    l2_values = tuple(
        float(value) for value in args.probe_l2_grid.split(",")
    )
    annotations = json.load(
        open(cfg["dataset"]["json_file"], encoding="utf-8")
    )["database"]

    source_train = make_dataset(
        cfg["dataset_name"], True, cfg["src_train_split"], **cfg["dataset"]
    )
    source_val = make_dataset(
        cfg["dataset_name"], True, cfg["src_val_split"], **cfg["dataset"]
    )
    target_train = make_dataset(
        cfg["dataset_name"], True, cfg["tgt_train_split"], **cfg["dataset"]
    )
    model = make_meta_arch(cfg["model_name"], cfg, device=args.device).to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    source_records, source_stats, prototype_sum, prototype_count = collect_records(
        model,
        source_train,
        annotations,
        cfg,
        "source",
        args.seed,
        batch_size,
        alpha,
        levels,
        collect_prototypes=True,
    )
    val_records, val_stats = collect_records(
        model,
        source_val,
        annotations,
        cfg,
        "source",
        args.seed + 1000,
        batch_size,
        alpha,
        levels,
    )
    target_records, target_stats = collect_records(
        model,
        target_train,
        annotations,
        cfg,
        "target",
        args.seed + 2000,
        batch_size,
        alpha,
        levels,
    )
    for records in (source_records, val_records, target_records):
        add_prototype_distance(records, prototype_sum, prototype_count)

    source_candidates = select_records(
        source_records, source_records["candidate"]
    )
    val_candidates = select_records(val_records, val_records["candidate"])
    target_candidates = select_records(
        target_records, target_records["candidate"]
    )

    output = {
        "audit": "source_only_action_reliability_probe",
        "config": args.config,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "seed": args.seed,
        "alpha": alpha,
        "model_mode": "eval",
        "label": "pseudo class matches target GT; GT background is incorrect",
        "target_gt_usage": (
            "offline labels and metrics only; never used for features, probe "
            "fitting, hyperparameters, or checkpoint selection"
        ),
        "probe_max_iter": args.probe_max_iter,
        "probe_l2_grid": list(l2_values),
        "levels": list(levels),
        "source_train": source_stats,
        "source_val": val_stats,
        "target_train": target_stats,
        "prototype_count": prototype_count.tolist(),
        "models": {},
        "separate_heads": {},
    }
    source_y = source_candidates["correct"].float()
    for name, feature_names in MODEL_FEATURES.items():
        train_x = feature_matrix(source_candidates, feature_names, n_levels)
        val_x = feature_matrix(val_candidates, feature_names, n_levels)
        target_x = feature_matrix(target_candidates, feature_names, n_levels)
        if name == "confidence":
            source_scores = source_candidates["score"]
            val_scores = val_candidates["score"]
            target_scores = target_candidates["score"]
            model_output = {"feature_names": ["score"], "kind": "raw_ranking"}
        else:
            probe, selected_l2, l2_selection = select_probe_by_source_val(
                train_x,
                source_y,
                val_x,
                val_candidates["correct"],
                l2_values,
                args.probe_max_iter,
            )
            source_scores = apply_probe(train_x, probe)
            val_scores = apply_probe(val_x, probe)
            target_scores = apply_probe(target_x, probe)
            model_output = serialize_probe(probe, feature_names)
            model_output["kind"] = "source_only_logistic_probe"
            model_output["selected_l2"] = selected_l2
            model_output["source_val_l2_selection"] = l2_selection
        model_output["source_train_metrics"] = evaluate_scores(
            source_scores, source_candidates
        )
        model_output["source_val_metrics"] = evaluate_scores(
            val_scores, val_candidates
        )
        model_output["target_train_metrics"] = evaluate_scores(
            target_scores, target_candidates
        )
        output["models"][name] = model_output

    for name, feature_names in MODEL_FEATURES.items():
        val_candidate_x = feature_matrix(
            val_candidates, feature_names, n_levels
        )
        target_candidate_x = feature_matrix(
            target_candidates, feature_names, n_levels
        )
        source_all_x = feature_matrix(source_records, feature_names, n_levels)
        if name == "confidence":
            val_foreground_scores = val_candidates["score"]
            target_foreground_scores = target_candidates["score"]
            val_class_scores = val_candidates["score"][
                val_candidates["foreground"]
            ]
            target_class_scores = target_candidates["score"][
                target_candidates["foreground"]
            ]
            val_joint_scores = val_candidates["score"]
            target_joint_scores = target_candidates["score"]
            head_output = {"kind": "raw_confidence_ranking"}
        else:
            foreground_probe, foreground_l2, foreground_selection = (
                select_probe_by_source_val(
                    source_all_x,
                    source_records["foreground"].float(),
                    val_candidate_x,
                    val_candidates["foreground"],
                    l2_values,
                    args.probe_max_iter,
                )
            )
            source_foreground = source_records["foreground"]
            val_foreground = val_candidates["foreground"]
            class_probe, class_l2, class_selection = select_probe_by_source_val(
                source_all_x[source_foreground],
                source_records["correct"][source_foreground].float(),
                val_candidate_x[val_foreground],
                val_candidates["correct"][val_foreground],
                l2_values,
                args.probe_max_iter,
            )
            val_foreground_logits = apply_probe(
                val_candidate_x, foreground_probe
            )
            target_foreground_logits = apply_probe(
                target_candidate_x, foreground_probe
            )
            val_class_logits = apply_probe(val_candidate_x, class_probe)
            target_class_logits = apply_probe(target_candidate_x, class_probe)
            val_foreground_scores = val_foreground_logits
            target_foreground_scores = target_foreground_logits
            val_class_scores = val_class_logits[val_foreground]
            target_class_scores = target_class_logits[
                target_candidates["foreground"]
            ]
            val_joint_scores = torch.sigmoid(val_foreground_logits) * torch.sigmoid(
                val_class_logits
            )
            target_joint_scores = torch.sigmoid(
                target_foreground_logits
            ) * torch.sigmoid(target_class_logits)
            head_output = {
                "kind": "two_independent_source_only_logistic_probes",
                "feature_names": list(feature_names),
                "foreground_probe": serialize_probe(
                    foreground_probe, feature_names
                ),
                "foreground_selected_l2": foreground_l2,
                "foreground_source_val_l2_selection": foreground_selection,
                "class_probe": serialize_probe(class_probe, feature_names),
                "class_selected_l2": class_l2,
                "class_source_val_l2_selection": class_selection,
            }
        head_output["source_val"] = {
            "foreground": evaluate_binary_scores(
                val_foreground_scores, val_candidates["foreground"]
            ),
            "class_given_foreground": evaluate_binary_scores(
                val_class_scores,
                val_candidates["correct"][val_candidates["foreground"]],
            ),
            "joint_correctness": evaluate_binary_scores(
                val_joint_scores, val_candidates["correct"]
            ),
        }
        head_output["target_train"] = {
            "foreground": evaluate_binary_scores(
                target_foreground_scores, target_candidates["foreground"]
            ),
            "class_given_foreground": evaluate_binary_scores(
                target_class_scores,
                target_candidates["correct"][target_candidates["foreground"]],
            ),
            "joint_correctness": evaluate_binary_scores(
                target_joint_scores, target_candidates["correct"]
            ),
        }
        output["separate_heads"][name] = head_output

    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    compact = {
        name: {
            "source_val_ap": values["source_val_metrics"]["average_precision"],
            "target_ap": values["target_train_metrics"]["average_precision"],
            "target_retain_060": values["target_train_metrics"]["retention"][
                "retain_060"
            ]["pseudo_correct_rate"],
        }
        for name, values in output["models"].items()
    }
    compact["separate_heads_target_ap"] = {
        name: {
            task: values["target_train"][task]["average_precision"]
            for task in (
                "foreground",
                "class_given_foreground",
                "joint_correctness",
            )
        }
        for name, values in output["separate_heads"].items()
    }
    print(json.dumps(compact, indent=2))
    print("wrote", args.output)


if __name__ == "__main__":
    main()

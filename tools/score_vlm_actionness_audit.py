"""Score confidence-conditioned VLM semantics against post-hoc domain GT."""

import argparse
import collections
import csv
import json
from pathlib import Path


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def safe_ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def normalize_verb(value):
    return str(value or "").strip().lower().replace(" ", "-").replace("_", "-")


def target_verbs_from_csv(path):
    with open(path, encoding="utf-8") as stream:
        return {
            normalize_verb(row["key"])
            for row in csv.DictReader(stream)
            if int(row["id"]) < 10
        }


def semantic_label(value):
    return {
        "A": "TARGET_ACTION",
        "B": "OTHER_ACTIVITY",
        "U": "UNCERTAIN",
    }.get(value, value)


def semantic_foreground(row):
    return bool(row.get("center_in_target_action", row["gt_is_foreground"]))


def semantic_gt_kind(row):
    if semantic_foreground(row):
        return "TARGET_ACTION"
    if row.get("center_all_gt_class_ids"):
        return "NON_TARGET_ACTIVITY"
    return "TRUE_BACKGROUND"


def summarize(rows):
    total = len(rows)
    foreground = sum(semantic_foreground(row) for row in rows)
    background = total - foreground
    predicted_action = sum(
        semantic_label(row["prediction"]) == "TARGET_ACTION" for row in rows
    )
    true_positive = sum(
        semantic_label(row["prediction"]) == "TARGET_ACTION"
        and semantic_foreground(row)
        for row in rows
    )
    false_positive = sum(
        semantic_label(row["prediction"]) == "TARGET_ACTION"
        and not semantic_foreground(row)
        for row in rows
    )
    correct_action = sum(
        semantic_label(row["prediction"]) == "TARGET_ACTION"
        and semantic_foreground(row)
        and row.get("sada_top1_is_correct", False)
        for row in rows
    )
    oracle_correct = sum(
        semantic_foreground(row) and row.get("sada_top1_is_correct", False)
        for row in rows
    )
    result = {
        "n": total,
        "gt_foreground_n": foreground,
        "gt_background_n": background,
        "prediction_counts": dict(
            collections.Counter(semantic_label(row["prediction"]) for row in rows)
        ),
        "candidate_foreground_recall": safe_ratio(true_positive, foreground),
        "background_false_promotion_rate": safe_ratio(false_positive, background),
        "promoted_action_precision": safe_ratio(true_positive, predicted_action),
        "uncertain_rate": safe_ratio(
            sum(semantic_label(row["prediction"]) == "UNCERTAIN" for row in rows),
            total,
        ),
    }
    if any("sada_top1_is_correct" in row for row in rows):
        result.update(
            {
                "correct_action_routing_coverage": safe_ratio(
                    correct_action, foreground
                ),
                "class_accuracy_given_recovered_foreground": safe_ratio(
                    correct_action, true_positive
                ),
                "oracle_actionness_correct_action_coverage": safe_ratio(
                    oracle_correct, foreground
                ),
            }
        )
    return result


def summarize_prompt_branch(rows, prompt_role):
    labels = [semantic_label(row["prediction"]) for row in rows]
    target = [semantic_foreground(row) for row in rows]
    label_counts = dict(collections.Counter(labels))
    result = {
        "n": len(rows),
        "gt_semantic_counts": dict(
            collections.Counter(semantic_gt_kind(row) for row in rows)
        ),
        "label_counts": label_counts,
    }
    if prompt_role == "low_rescue":
        passed = [label == "TARGET_ACTION" for label in labels]
        true_pass = sum(keep and truth for keep, truth in zip(passed, target))
        result.update(
            {
                "true_action_rescue_recall": safe_ratio(true_pass, sum(target)),
                "semantic_rescue_precision": safe_ratio(true_pass, sum(passed)),
                "pass_coverage": safe_ratio(sum(passed), len(rows)),
            }
        )
    elif prompt_role == "high_veto":
        vetoed = [
            label in {"OTHER_ACTIVITY", "TRUE_BACKGROUND"} for label in labels
        ]
        negatives = [not truth for truth in target]
        true_veto = sum(veto and negative for veto, negative in zip(vetoed, negatives))
        target_kept = sum((not veto) and truth for veto, truth in zip(vetoed, target))
        non_target = [semantic_gt_kind(row) == "NON_TARGET_ACTIVITY" for row in rows]
        true_background = [semantic_gt_kind(row) == "TRUE_BACKGROUND" for row in rows]
        result.update(
            {
                "true_action_retention": safe_ratio(target_kept, sum(target)),
                "background_non_target_rejection": safe_ratio(
                    true_veto, sum(negatives)
                ),
                "non_target_activity_rejection": safe_ratio(
                    sum(veto and truth for veto, truth in zip(vetoed, non_target)),
                    sum(non_target),
                ),
                "true_background_rejection": safe_ratio(
                    sum(
                        veto and truth
                        for veto, truth in zip(vetoed, true_background)
                    ),
                    sum(true_background),
                ),
                "veto_precision": safe_ratio(true_veto, sum(vetoed)),
                "veto_coverage": safe_ratio(sum(vetoed), len(rows)),
            }
        )
    else:
        raise ValueError("unsupported prompt role: {}".format(prompt_role))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--metadata",
        help="Exporter sidecar; defaults to <manifest>.meta.json when present.",
    )
    parser.add_argument(
        "--ontology-csv",
        help="Derive A/B deterministically from reported_verb and the first ten IDs.",
    )
    args = parser.parse_args()

    manifest = {row["candidate_id"]: row for row in read_jsonl(args.manifest)}
    predictions = {
        row["candidate_id"]: row
        for row in read_jsonl(args.predictions)
        if row.get("status") == "ok"
    }
    cached_gate_rules = {
        row.get("gate_rule") for row in predictions.values() if row.get("gate_rule")
    }
    target_verbs = target_verbs_from_csv(args.ontology_csv) if args.ontology_csv else None
    rows = []
    for candidate_id, prediction in predictions.items():
        if candidate_id not in manifest:
            continue
        row = {**manifest[candidate_id], **prediction}
        if target_verbs is not None and row.get("reported_verb"):
            row["model_reported_prediction"] = row["prediction"]
            verb = normalize_verb(row["reported_verb"])
            row["prediction"] = "U" if verb == "uncertain" else (
                "A" if verb in target_verbs else "B"
            )
        rows.append(row)

    by_level = {}
    for level in sorted({row["fpn_level"] for row in rows if "fpn_level" in row}):
        by_level[str(level)] = summarize(
            [row for row in rows if row["fpn_level"] == level]
        )
    by_probe_group = {}
    for group in sorted({row["probe_group"] for row in rows if "probe_group" in row}):
        by_probe_group[group] = summarize(
            [row for row in rows if row["probe_group"] == group]
        )
    by_prompt_role = {}
    for role in sorted({row.get("prompt_role", "low_rescue") for row in rows}):
        by_prompt_role[role] = summarize_prompt_branch(
            [row for row in rows if row.get("prompt_role", "low_rescue") == role],
            role,
        )
    metadata_path = Path(args.metadata or (args.manifest + ".meta.json"))
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else None
    )
    global_metrics = summarize(rows)
    if metadata is not None and "gt_foreground_anchors" in metadata:
        promoted_foreground = sum(
            semantic_label(row["prediction"]) == "TARGET_ACTION"
            and row["gt_is_foreground"]
            for row in rows
        )
        promoted_correct = sum(
            semantic_label(row["prediction"]) == "TARGET_ACTION"
            and row["gt_is_foreground"]
            and row["sada_top1_is_correct"]
            for row in rows
        )
        denominator = metadata["gt_foreground_anchors"]
        global_metrics["foreground_routing_coverage_before_vlm"] = safe_ratio(
            metadata["original_selected_foreground_anchors"], denominator
        )
        global_metrics["foreground_routing_coverage_after_vlm"] = safe_ratio(
            metadata["original_selected_foreground_anchors"] + promoted_foreground,
            denominator,
        )
        global_metrics["correct_action_routing_coverage_before_vlm"] = safe_ratio(
            metadata["original_selected_correct_action_anchors"], denominator
        )
        global_metrics["correct_action_routing_coverage_after_vlm"] = safe_ratio(
            metadata["original_selected_correct_action_anchors"] + promoted_correct,
            denominator,
        )

    result = {
        "audit": "offline_vlm_actionness",
        "domain": metadata.get("domain") if metadata else None,
        "gt_use": "post-API scoring only; never included in prompt",
        "gate_rule": (
            "deterministic_from_reported_verb"
            if target_verbs is not None
            else (
                next(iter(cached_gate_rules))
                if len(cached_gate_rules) == 1
                else "cached_model_label"
            )
        ),
        "completed_candidates": len(rows),
        "missing_candidates": len(manifest) - len(rows),
        "global": global_metrics,
        "by_fpn_level": by_level,
        "by_probe_group": by_probe_group,
        "by_prompt_role": by_prompt_role,
        "candidate_export_metadata": metadata,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

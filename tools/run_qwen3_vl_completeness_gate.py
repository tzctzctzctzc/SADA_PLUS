"""Score proposal completeness with explicit before/inside/after Qwen evidence.

This is a source-only gate.  It asks Qwen for a fixed boundary-status option,
then checks whether those scores add proposal-IoU information beyond SlowFast.
Target GT is never read by this program.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from tools.evaluate_qwen3_vl_proposal_features import (
    _rank_metrics,
    group_ridge_predictions,
    proposal_quality_targets,
    slowfast_variants,
)
from tools.run_clip_then_llava import frame_paths, read_jsonl, write_json
from tools.run_qwen3_vl_classification import load_qwen3


OUTCOMES = (
    "WELL_ALIGNED",
    "STARTS_TOO_LATE",
    "ENDS_TOO_EARLY",
    "TOO_WIDE_OR_MULTIPLE",
    "NO_SUPPORTED_KNOWN_ACTION",
)
PROMPT_VERSION = "qwen3-vl-before-inside-after-completeness-v1"


def outcome_mapping(candidate_id):
    """Rotate answer positions deterministically to reduce digit-position bias."""
    offset = int.from_bytes(
        hashlib.sha256(str(candidate_id).encode("utf-8")).digest()[:2], "big"
    ) % len(OUTCOMES)
    return OUTCOMES[offset:] + OUTCOMES[:offset]


def build_prompt(candidate_id):
    descriptions = {
        "WELL_ALIGNED": (
            "INSIDE contains one complete benchmark action and its start and end are "
            "close to the INSIDE boundaries"
        ),
        "STARTS_TOO_LATE": (
            "the action is already underway in BEFORE, so INSIDE misses its beginning"
        ),
        "ENDS_TOO_EARLY": (
            "the action continues into AFTER, so INSIDE misses its ending"
        ),
        "TOO_WIDE_OR_MULTIPLE": (
            "INSIDE contains substantial unrelated activity, idle time, or multiple actions"
        ),
        "NO_SUPPORTED_KNOWN_ACTION": (
            "there is no clearly supported complete action from the benchmark set"
        ),
    }
    mapping = outcome_mapping(candidate_id)
    options = "\n".join(
        "{}: {} ({})".format(index, outcome, descriptions[outcome])
        for index, outcome in enumerate(mapping)
    )
    return (
        "You are judging the temporal boundaries of one candidate action proposal. "
        "Frames 1-4 are BEFORE the candidate, frames 5-12 are INSIDE it, and "
        "frames 13-16 are AFTER it. The benchmark actions are TAKE, PUT, WASH, "
        "OPEN, CLOSE, INSERT, TURN_ON, CUT, TURN_OFF, and POUR. Track the main "
        "hand-object state across all three regions. Judge boundary completeness, "
        "not just whether an action-looking object is visible. Choose exactly one:\n"
        + options
        + "\nAnswer immediately with exactly one digit from 0 to 4."
    )


def prompt_sha256(candidate_id):
    return hashlib.sha256(build_prompt(candidate_id).encode("utf-8")).hexdigest()


def build_messages(paths, prompt, min_pixels, max_pixels):
    if len(paths) != 16:
        raise ValueError("completeness prompt requires exactly 16 frames")
    content = []
    for index, path in enumerate(paths, 1):
        if index == 1:
            content.append({"type": "text", "text": "BEFORE region:"})
        elif index == 5:
            content.append({"type": "text", "text": "INSIDE candidate region:"})
        elif index == 13:
            content.append({"type": "text", "text": "AFTER region:"})
        content.append({"type": "text", "text": "FRAME {}".format(index)})
        content.append({
            "type": "image",
            "image": Path(path).resolve().as_uri(),
            "min_pixels": int(min_pixels),
            "max_pixels": int(max_pixels),
        })
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def normalize_logits(logits, mapping):
    maximum = max(float(value) for value in logits)
    weights = [math.exp(float(value) - maximum) for value in logits]
    total = sum(weights)
    digit_probabilities = [value / total for value in weights]
    probabilities = {
        outcome: digit_probabilities[index]
        for index, outcome in enumerate(mapping)
    }
    prediction = max(probabilities, key=probabilities.get)
    return probabilities, prediction


def infer_one(runtime, paths, candidate_id, args):
    torch = runtime["torch"]
    prompt = build_prompt(candidate_id)
    mapping = outcome_mapping(candidate_id)
    messages = build_messages(paths, prompt, args.min_pixels, args.max_pixels)
    text = runtime["processor"].apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = runtime["process_vision_info"](
        messages, image_patch_size=16
    )
    inputs = runtime["processor"](
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(runtime["device"])
    with torch.inference_mode():
        output = runtime["model"](
            **inputs, use_cache=False, logits_to_keep=1, return_dict=True
        )
    next_logits = output.logits[0, -1].float()
    token_ids = torch.tensor(
        runtime["digit_token_ids"][:len(OUTCOMES)],
        device=next_logits.device,
        dtype=torch.long,
    )
    logits = next_logits.index_select(0, token_ids).detach().cpu().tolist()
    probabilities, prediction = normalize_logits(logits, mapping)
    return {
        "prediction": prediction,
        "outcome_probabilities": probabilities,
        "digit_logits": logits,
        "digit_to_outcome": list(mapping),
        "prompt": prompt,
        "prompt_sha256": prompt_sha256(candidate_id),
    }


def prediction_is_current(row, args):
    return bool(
        row.get("status") == "ok"
        and row.get("model") == args.model_path
        and row.get("prompt_version") == PROMPT_VERSION
        and row.get("prompt_sha256") == prompt_sha256(row.get("candidate_id"))
    )


def score_manifest(runtime, args):
    rows = read_jsonl(args.manifest)
    completed = {
        row["candidate_id"]
        for row in read_jsonl(args.output)
        if prediction_is_current(row, args)
    }
    pending = [row for row in rows if row["candidate_id"] not in completed]
    if args.limit > 0:
        pending = pending[:args.limit]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for number, row in enumerate(pending, 1):
        try:
            paths = frame_paths(
                args.frame_cache, args.cache_namespace, row["candidate_id"],
                frame_count=16,
            )
            result = {
                **row,
                **infer_one(runtime, paths, row["candidate_id"], args),
                "status": "ok",
                "model": args.model_path,
                "model_revision": runtime["model_revision"],
                "prompt_version": PROMPT_VERSION,
                "frame_count": 16,
            }
        except Exception as error:
            result = {
                "candidate_id": row["candidate_id"],
                "video_id": row.get("video_id"),
                "status": "error",
                "error": "{}: {}".format(type(error).__name__, error),
            }
        with output_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result, sort_keys=True) + "\n")
        print(
            "{}/{} {} {}".format(
                number, len(pending), row["candidate_id"], result["status"]
            ),
            flush=True,
        )


def evaluate_gate(args):
    manifest = read_jsonl(args.manifest)
    database = json.loads(Path(args.annotation_json).read_text(encoding="utf-8"))[
        "database"
    ]
    rows = proposal_quality_targets(manifest, database)
    predictions = {
        row["candidate_id"]: row
        for row in read_jsonl(args.output)
        if prediction_is_current(row, args)
    }
    slowfast = slowfast_variants(
        rows, args.slowfast_root, args.fps, args.feat_stride, args.feat_num_frames
    )["slowfast_combined"]
    rows = [
        row for row in rows
        if row["candidate_id"] in predictions and row["candidate_id"] in slowfast
    ]
    groups = np.asarray([row["video_id"] for row in rows])
    structure = np.asarray([
        [
            float(row["teacher_score"]),
            np.log1p(float(row["proposal_end_sec"]) - float(row["proposal_start_sec"])),
            *[float(int(row["source_fpn_levels"][0]) == level) for level in range(3)],
        ]
        for row in rows
    ], dtype=np.float32)
    sf = np.asarray([slowfast[row["candidate_id"]] for row in rows])
    prompt = np.asarray([
        [predictions[row["candidate_id"]]["outcome_probabilities"][name]
         for name in OUTCOMES]
        for row in rows
    ], dtype=np.float32)

    report = {
        "protocol_version": PROMPT_VERSION,
        "selection_split": "source_val",
        "target_gt_usage": "forbidden",
        "count": len(rows),
        "prompt_sha256": hashlib.sha256(
            json.dumps(
                [build_prompt("protocol-template")], sort_keys=True
            ).encode("utf-8")
        ).hexdigest(),
        "targets": {},
    }
    feature_sets = {
        "structure_slowfast": {"structure": structure, "slowfast": sf},
        "structure_slowfast_prompt": {
            "structure": structure, "slowfast": sf, "prompt": prompt
        },
        "structure_prompt": {"structure": structure, "prompt": prompt},
    }
    for target_name in ("quality_iou_any", "quality_iou_class"):
        targets = np.asarray([row[target_name] for row in rows], dtype=np.float32)
        methods = {
            "qwen_well_aligned": _rank_metrics(
                targets, prompt[:, OUTCOMES.index("WELL_ALIGNED")]
            )
        }
        for name, blocks in feature_sets.items():
            scores = group_ridge_predictions(blocks, targets, groups, args.folds)
            methods[name] = _rank_metrics(targets, scores)
        baseline = methods["structure_slowfast"]
        fusion = methods["structure_slowfast_prompt"]
        report["targets"][target_name] = {
            "methods": methods,
            "fusion_vs_slowfast": {
                "auc_gain": (
                    fusion["roc_auc_iou_0.5"] - baseline["roc_auc_iou_0.5"]
                ),
                "spearman_gain": fusion["spearman_iou"] - baseline["spearman_iou"],
            },
        }
    evidence = report["targets"]["quality_iou_any"]["fusion_vs_slowfast"]
    report["thresholds"] = {
        "min_auc_gain": args.min_auc_gain,
        "min_spearman_gain": args.min_spearman_gain,
    }
    report["source_gate_pass"] = bool(
        evidence["auc_gain"] >= args.min_auc_gain
        and evidence["spearman_gain"] >= args.min_spearman_gain
    )
    write_json(args.gate_output, report)
    return report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--frame-cache", required=True)
    parser.add_argument("--cache-namespace", default="s5_source")
    parser.add_argument("--output", required=True)
    parser.add_argument("--gate-output", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--slowfast-root", required=True)
    parser.add_argument(
        "--annotation-json",
        default="data/ek100_slowfast/annotations/epic_kitchens_100_verb_detailed_split.json",
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--device-map", choices=("none", "auto"), default="none")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument(
        "--dtype", choices=("auto", "bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument("--min-pixels", type=int, default=224 * 224)
    parser.add_argument("--max-pixels", type=int, default=448 * 448)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="reuse completed predictions and run only the source quality gate",
    )
    parser.add_argument(
        "--inference-only",
        action="store_true",
        help="write predictions without importing source-gate dependencies",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--feat-stride", type=float, default=16.0)
    parser.add_argument("--feat-num-frames", type=float, default=32.0)
    parser.add_argument("--min-auc-gain", type=float, default=0.02)
    parser.add_argument("--min-spearman-gain", type=float, default=0.03)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.score_only:
        runtime = load_qwen3(args)
        score_manifest(runtime, args)
    if args.inference_only:
        return
    report = evaluate_gate(args)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

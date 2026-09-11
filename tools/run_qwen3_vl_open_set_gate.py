"""Run a source-only KNOWN/OTHER/BACKGROUND gate with Qwen3-VL."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from tools.evaluate_qwen3_vl_proposal_features import (
    _rank_metrics,
    group_ridge_predictions,
    slowfast_variants,
)
from tools.run_clip_then_llava import frame_paths, read_jsonl, write_json
from tools.run_qwen3_vl_classification import build_messages, load_qwen3


OUTCOMES = ("KNOWN_ACTION", "OTHER_ACTION", "BACKGROUND_OR_UNCERTAIN")
PROMPT_VERSION = "qwen3-vl-open-set-actionness-v1"


def outcome_mapping(candidate_id):
    offset = int.from_bytes(
        hashlib.sha256(str(candidate_id).encode("utf-8")).digest()[:2], "big"
    ) % len(OUTCOMES)
    return OUTCOMES[offset:] + OUTCOMES[:offset]


def build_prompt(candidate_id, frame_count=8):
    mapping = outcome_mapping(candidate_id)
    descriptions = {
        "KNOWN_ACTION": (
            "the proposal visibly supports one of TAKE, PUT, WASH, OPEN, CLOSE, "
            "INSERT, TURN_ON, CUT, TURN_OFF, or POUR"
        ),
        "OTHER_ACTION": (
            "a deliberate hand-object action is visible, but it is not one of those ten"
        ),
        "BACKGROUND_OR_UNCERTAIN": (
            "no deliberate action is clearly supported, or the evidence is insufficient"
        ),
    }
    options = "\n".join(
        "{}: {} ({})".format(index, outcome, descriptions[outcome])
        for index, outcome in enumerate(mapping)
    )
    return (
        "The {} images are chronological frames sampled inside one temporal proposal. "
        "Decide whether the visible evidence belongs to the benchmark action set. "
        "Do not infer an unseen intention from the scene or object identity alone.\n{}\n"
        "Answer immediately with exactly one digit from 0 to 2."
    ).format(frame_count, options)


def prompt_sha256(candidate_id, frame_count=8):
    return hashlib.sha256(
        build_prompt(candidate_id, frame_count).encode("utf-8")
    ).hexdigest()


def normalize_logits(logits, mapping):
    maximum = max(float(value) for value in logits)
    weights = [math.exp(float(value) - maximum) for value in logits]
    total = sum(weights)
    probabilities = {
        outcome: weights[index] / total for index, outcome in enumerate(mapping)
    }
    return probabilities, max(probabilities, key=probabilities.get)


def infer_one(runtime, paths, candidate_id, args):
    torch = runtime["torch"]
    prompt = build_prompt(candidate_id, args.frame_count)
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
    logits = output.logits[0, -1].float()
    token_ids = torch.tensor(
        runtime["digit_token_ids"][:len(OUTCOMES)],
        device=logits.device,
        dtype=torch.long,
    )
    digit_logits = logits.index_select(0, token_ids).detach().cpu().tolist()
    probabilities, prediction = normalize_logits(digit_logits, mapping)
    return {
        "prediction": prediction,
        "outcome_probabilities": probabilities,
        "digit_logits": digit_logits,
        "digit_to_outcome": list(mapping),
        "prompt_sha256": prompt_sha256(candidate_id, args.frame_count),
    }


def prediction_is_current(row, args):
    return bool(
        row.get("status") == "ok"
        and row.get("model") == args.model_path
        and row.get("prompt_version") == PROMPT_VERSION
        and row.get("frame_count") == args.frame_count
        and row.get("prompt_sha256")
        == prompt_sha256(row.get("candidate_id"), args.frame_count)
    )


def score_manifest(runtime, args):
    rows = read_jsonl(args.manifest)
    completed = {
        row["candidate_id"] for row in read_jsonl(args.output)
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
                frame_count=args.frame_count,
            )
            result = {
                **row,
                **infer_one(runtime, paths, row["candidate_id"], args),
                "status": "ok",
                "model": args.model_path,
                "model_revision": runtime["model_revision"],
                "prompt_version": PROMPT_VERSION,
                "frame_count": args.frame_count,
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
    predictions = {
        row["candidate_id"]: row for row in read_jsonl(args.output)
        if prediction_is_current(row, args)
    }
    slowfast = slowfast_variants(
        manifest, args.slowfast_root, args.fps, args.feat_stride,
        args.feat_num_frames,
    )["slowfast_combined"]
    rows = [
        row for row in manifest
        if row["candidate_id"] in predictions and row["candidate_id"] in slowfast
    ]
    targets = np.asarray([
        row.get("semantic_gt_kind") == "KNOWN_ACTION" for row in rows
    ], dtype=np.float32)
    groups = np.asarray([row["video_id"] for row in rows])
    structure = np.asarray([
        [
            float(row["teacher_score"]),
            np.log1p(float(row["proposal_end_sec"]) - float(row["proposal_start_sec"])),
            *[float(int(row["source_fpn_levels"][0]) == level) for level in range(3)],
        ] for row in rows
    ], dtype=np.float32)
    sf = np.asarray([slowfast[row["candidate_id"]] for row in rows])
    prompt = np.asarray([
        [predictions[row["candidate_id"]]["outcome_probabilities"][name]
         for name in OUTCOMES]
        for row in rows
    ], dtype=np.float32)
    direct = _rank_metrics(
        targets, prompt[:, OUTCOMES.index("KNOWN_ACTION")], threshold=0.5
    )
    teacher = _rank_metrics(
        targets, [row["teacher_score"] for row in rows], threshold=0.5
    )
    baseline_scores = group_ridge_predictions(
        {"structure": structure, "slowfast": sf}, targets, groups, args.folds
    )
    fusion_scores = group_ridge_predictions(
        {"structure": structure, "slowfast": sf, "prompt": prompt},
        targets, groups, args.folds,
    )
    baseline = _rank_metrics(targets, baseline_scores, threshold=0.5)
    fusion = _rank_metrics(targets, fusion_scores, threshold=0.5)
    counts = {}
    for row in rows:
        name = predictions[row["candidate_id"]]["prediction"]
        counts[name] = counts.get(name, 0) + 1
    report = {
        "protocol_version": PROMPT_VERSION,
        "selection_split": "source_val",
        "target_gt_usage": "forbidden",
        "count": len(rows),
        "known_count": int(targets.sum()),
        "prediction_counts": counts,
        "methods": {
            "qwen_known_probability": direct,
            "teacher_score": teacher,
            "structure_slowfast": baseline,
            "structure_slowfast_prompt": fusion,
        },
        "fusion_vs_slowfast": {
            "auc_gain": fusion["roc_auc_iou_0.5"] - baseline["roc_auc_iou_0.5"],
            "ap_gain": (
                fusion["average_precision_iou_0.5"]
                - baseline["average_precision_iou_0.5"]
            ),
        },
        "thresholds": {
            "min_auc_gain": args.min_auc_gain,
            "min_ap_gain": args.min_ap_gain,
        },
    }
    gain = report["fusion_vs_slowfast"]
    report["source_gate_pass"] = bool(
        gain["auc_gain"] >= args.min_auc_gain
        and gain["ap_gain"] >= args.min_ap_gain
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
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--device-map", choices=("none", "auto"), default="none")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument(
        "--dtype", choices=("auto", "bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument("--frame-count", type=int, default=8)
    parser.add_argument("--min-pixels", type=int, default=224 * 224)
    parser.add_argument("--max-pixels", type=int, default=448 * 448)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--feat-stride", type=float, default=16.0)
    parser.add_argument("--feat-num-frames", type=float, default=32.0)
    parser.add_argument("--min-auc-gain", type=float, default=0.02)
    parser.add_argument("--min-ap-gain", type=float, default=0.02)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.score_only:
        score_manifest(load_qwen3(args), args)
    report = evaluate_gate(args)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

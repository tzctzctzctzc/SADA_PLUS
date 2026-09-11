"""Source-gated Qwen state expert for PUT / INSERT / POUR confusions.

The expert is queried only when the frozen TAL teacher or the frozen ten-class
Qwen classifier predicts one of the three routed classes.  Its four-way logits
are converted to likelihood ratios against OTHER and added to the original
ten-class logits.  All routing and fusion weights are selected on source data.
"""

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

from tools.run_clip_target_adapter import LABELS
from tools.run_clip_then_llava import frame_paths, read_jsonl, write_json
from tools.run_qwen3_vl_classification import build_messages, load_qwen3


EXPERT_LABELS = ("PUT", "INSERT", "POUR", "OTHER")
ROUTED_CLASS_IDS = tuple(LABELS.index(label) for label in EXPERT_LABELS[:3])
PROMPT_VERSION = "qwen3-vl-put-insert-pour-state-expert-v1"
WIRE_API = "qwen3-vl-next-token-state-expert"
DEFAULT_ALPHAS = (0.0, 0.25, 0.5, 1.0, 2.0)


def option_mapping(candidate_id):
    seed = int.from_bytes(
        hashlib.sha256(str(candidate_id).encode("utf-8")).digest()[:8], "big"
    )
    labels = list(EXPERT_LABELS)
    random.Random(seed).shuffle(labels)
    return tuple(labels)


def build_prompt(candidate_id, frame_count=8):
    mapping = option_mapping(candidate_id)
    descriptions = {
        "PUT": (
            "a solid object starts held or moving and ends resting or released at a "
            "location; there is no visible flowing material and no distinct fitted "
            "entry through a narrow opening"
        ),
        "INSERT": (
            "a solid object visibly passes from outside through a distinct slot, "
            "socket, hole, or constrained opening and ends inside or fitted into it"
        ),
        "POUR": (
            "a source vessel tilts and its contents visibly flow or transfer into "
            "another place; moving or tilting the vessel alone is insufficient"
        ),
        "OTHER": (
            "the visible action is another action such as TAKE, OPEN, CLOSE, WASH, "
            "TURN_ON, CUT, or TURN_OFF, or the evidence is too incomplete to support "
            "PUT, INSERT, or POUR"
        ),
    }
    options = "\n".join(
        "{}: {} ({})".format(index, label, descriptions[label])
        for index, label in enumerate(mapping)
    )
    return (
        "The {} images are chronological frames sampled inside one temporal proposal. "
        "Judge the visible state transition, not the object identity or an unseen "
        "intention. Compare the object's state in the early and late frames. Treat "
        "opening or closing a container as incidental when the main transition is "
        "placing or inserting another object. Choose exactly one option.\n{}\n"
        "Answer immediately with exactly one digit from 0 to 3."
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
        label: weights[index] / total for index, label in enumerate(mapping)
    }
    return probabilities, max(probabilities, key=probabilities.get)


def baseline_topk(row, k=1):
    probabilities = row.get("class_probabilities")
    if probabilities is None or len(probabilities) != len(LABELS):
        raise ValueError("baseline row must contain ten class probabilities")
    return sorted(
        range(len(probabilities)),
        key=lambda index: float(probabilities[index]),
        reverse=True,
    )[:k]


def should_route(row, topk=1):
    teacher = int(row.get("teacher_top1_class", -1))
    candidates = set(baseline_topk(row, topk))
    return teacher in ROUTED_CLASS_IDS or bool(candidates.intersection(ROUTED_CLASS_IDS))


def fuse_logits(row, expert_probabilities, alpha):
    logits = [float(value) for value in row["class_logits"]]
    other = max(float(expert_probabilities["OTHER"]), 1e-8)
    for label in EXPERT_LABELS[:3]:
        class_id = LABELS.index(label)
        evidence = math.log(max(float(expert_probabilities[label]), 1e-8) / other)
        logits[class_id] += float(alpha) * evidence
    return logits


def predicted_class(logits):
    return max(range(len(logits)), key=lambda index: logits[index])


def source_accuracy(rows, expert_by_id, alpha, route_topk=1):
    known = [
        row for row in rows
        if len(row.get("gt_assigned_class_ids") or []) == 1
        and int(row["gt_assigned_class_ids"][0]) < len(LABELS)
    ]
    routed = [row for row in known if should_route(row, route_topk)]

    def correct(row, use_expert):
        logits = row["class_logits"]
        if use_expert:
            expert = expert_by_id.get(row["candidate_id"])
            if expert is not None:
                logits = fuse_logits(row, expert["expert_probabilities"], alpha)
        return predicted_class(logits) == int(row["gt_assigned_class_ids"][0])

    overall_correct = sum(correct(row, True) for row in known)
    routed_correct = sum(correct(row, True) for row in routed)
    baseline_overall = sum(correct(row, False) for row in known)
    baseline_routed = sum(correct(row, False) for row in routed)
    return {
        "known_count": len(known),
        "routed_known_count": len(routed),
        "overall_correct": overall_correct,
        "routed_correct": routed_correct,
        "baseline_overall_correct": baseline_overall,
        "baseline_routed_correct": baseline_routed,
        "overall_accuracy": overall_correct / len(known) if known else None,
        "routed_accuracy": routed_correct / len(routed) if routed else None,
        "baseline_overall_accuracy": baseline_overall / len(known) if known else None,
        "baseline_routed_accuracy": baseline_routed / len(routed) if routed else None,
    }


def select_alpha(rows, expert_by_id, alphas, route_topk=1):
    reports = {
        str(alpha): source_accuracy(rows, expert_by_id, alpha, route_topk)
        for alpha in alphas
    }
    best = max(
        alphas,
        key=lambda alpha: (
            reports[str(alpha)]["routed_accuracy"] or -1.0,
            reports[str(alpha)]["overall_accuracy"] or -1.0,
            -float(alpha),
        ),
    )
    return float(best), reports


def cross_validated_accuracy(rows, expert_by_id, alphas, route_topk=1):
    videos = sorted({
        row["video_id"] for row in rows
        if len(row.get("gt_assigned_class_ids") or []) == 1
        and int(row["gt_assigned_class_ids"][0]) < len(LABELS)
    })
    totals = {
        "known_count": 0,
        "routed_known_count": 0,
        "overall_correct": 0,
        "routed_correct": 0,
        "baseline_overall_correct": 0,
        "baseline_routed_correct": 0,
    }
    fold_alphas = {}
    for video_id in videos:
        train = [row for row in rows if row.get("video_id") != video_id]
        test = [row for row in rows if row.get("video_id") == video_id]
        alpha, _ = select_alpha(train, expert_by_id, alphas, route_topk)
        fold_alphas[video_id] = alpha
        fold = source_accuracy(test, expert_by_id, alpha, route_topk)
        for key in totals:
            totals[key] += int(fold[key])
    totals.update({
        "overall_accuracy": (
            totals["overall_correct"] / totals["known_count"]
            if totals["known_count"] else None
        ),
        "routed_accuracy": (
            totals["routed_correct"] / totals["routed_known_count"]
            if totals["routed_known_count"] else None
        ),
        "baseline_overall_accuracy": (
            totals["baseline_overall_correct"] / totals["known_count"]
            if totals["known_count"] else None
        ),
        "baseline_routed_accuracy": (
            totals["baseline_routed_correct"] / totals["routed_known_count"]
            if totals["routed_known_count"] else None
        ),
        "fold_selected_alphas": fold_alphas,
    })
    return totals


def infer_one(runtime, paths, candidate_id, args):
    torch = runtime["torch"]
    prompt = build_prompt(candidate_id, args.frame_count)
    mapping = option_mapping(candidate_id)
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
        runtime["digit_token_ids"][:len(EXPERT_LABELS)],
        device=logits.device,
        dtype=torch.long,
    )
    digit_logits = logits.index_select(0, token_ids).detach().cpu().tolist()
    probabilities, prediction = normalize_logits(digit_logits, mapping)
    return {
        "expert_prediction": prediction,
        "expert_probabilities": probabilities,
        "digit_logits": digit_logits,
        "digit_to_expert_label": list(mapping),
        "prompt_sha256": prompt_sha256(candidate_id, args.frame_count),
    }


def prediction_is_current(row, args):
    return bool(
        row.get("status") == "ok"
        and row.get("model") == args.model_path
        and row.get("prompt_version") == PROMPT_VERSION
        and row.get("frame_count") == args.frame_count
        and row.get("route_topk") == args.route_topk
        and row.get("prompt_sha256")
        == prompt_sha256(row.get("candidate_id"), args.frame_count)
    )


def score_manifest(runtime, args):
    rows = read_jsonl(args.baseline_predictions)
    routed = [row for row in rows if should_route(row, args.route_topk)]
    completed = {
        row["candidate_id"] for row in read_jsonl(args.output)
        if prediction_is_current(row, args)
    }
    pending = [row for row in routed if row["candidate_id"] not in completed]
    if args.limit > 0:
        pending = pending[:args.limit]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for number, row in enumerate(pending, 1):
        try:
            paths = frame_paths(
                args.frame_cache, args.cache_namespace, row["candidate_id"],
                frame_count=args.frame_count,
            )
            result = {
                **infer_one(runtime, paths, row["candidate_id"], args),
                "candidate_id": row["candidate_id"],
                "video_id": row.get("video_id"),
                "status": "ok",
                "model": args.model_path,
                "model_revision": runtime["model_revision"],
                "prompt_version": PROMPT_VERSION,
                "wire_api": WIRE_API,
                "frame_count": args.frame_count,
                "route_topk": args.route_topk,
            }
        except Exception as error:
            result = {
                "candidate_id": row["candidate_id"],
                "video_id": row.get("video_id"),
                "status": "error",
                "error": "{}: {}".format(type(error).__name__, error),
            }
        with output.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result, sort_keys=True) + "\n")
        print(
            "{}/{} {} {}".format(
                number, len(pending), row["candidate_id"], result["status"]
            ),
            flush=True,
        )


def evaluate_gate(args):
    rows = read_jsonl(args.baseline_predictions)
    expert_by_id = {
        row["candidate_id"]: row for row in read_jsonl(args.output)
        if prediction_is_current(row, args)
    }
    routed_count = sum(should_route(row, args.route_topk) for row in rows)
    success_rate = len(expert_by_id) / routed_count if routed_count else 0.0
    best_alpha, reports = select_alpha(
        rows, expert_by_id, args.alphas, args.route_topk
    )
    best = reports[str(best_alpha)]
    cross_validated = cross_validated_accuracy(
        rows, expert_by_id, args.alphas, args.route_topk
    )
    routed_gain = (
        cross_validated["routed_accuracy"]
        - cross_validated["baseline_routed_accuracy"]
    )
    overall_gain = (
        cross_validated["overall_accuracy"]
        - cross_validated["baseline_overall_accuracy"]
    )
    report = {
        "protocol_version": PROMPT_VERSION,
        "selection_split": "source_val",
        "target_gt_usage": "forbidden",
        "route": "teacher_top1_or_qwen_top{}_in_put_insert_pour".format(
            args.route_topk
        ),
        "routed_count": routed_count,
        "success_rate": success_rate,
        "selected_alpha": best_alpha,
        "alpha_reports": reports,
        "source_fit": best,
        "cross_validated": cross_validated,
        "cross_validated_routed_accuracy_gain": routed_gain,
        "cross_validated_overall_accuracy_gain": overall_gain,
        "thresholds": {
            "min_success_rate": args.min_success_rate,
            "min_routed_accuracy_gain": args.min_routed_accuracy_gain,
            "min_overall_accuracy_gain": args.min_overall_accuracy_gain,
        },
    }
    report["source_gate_pass"] = bool(
        success_rate >= args.min_success_rate
        and best_alpha > 0.0
        and routed_gain >= args.min_routed_accuracy_gain
        and overall_gain >= args.min_overall_accuracy_gain
    )
    write_json(args.gate_output, report)
    return report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-predictions", required=True)
    parser.add_argument("--frame-cache", required=True)
    parser.add_argument("--cache-namespace", default="s5_source")
    parser.add_argument("--output", required=True)
    parser.add_argument("--gate-output", required=True)
    parser.add_argument("--model-path", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--device-map", choices=("none", "auto"), default="none")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument(
        "--dtype", choices=("auto", "bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument("--frame-count", type=int, default=8)
    parser.add_argument("--route-topk", type=int, default=1)
    parser.add_argument("--min-pixels", type=int, default=224 * 224)
    parser.add_argument("--max-pixels", type=int, default=448 * 448)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS)
    )
    parser.add_argument("--min-success-rate", type=float, default=0.95)
    parser.add_argument("--min-routed-accuracy-gain", type=float, default=0.03)
    parser.add_argument("--min-overall-accuracy-gain", type=float, default=0.01)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 1 <= args.route_topk <= len(LABELS):
        raise ValueError("route top-k must be between 1 and ten")
    runtime = load_qwen3(args)
    score_manifest(runtime, args)
    report = evaluate_gate(args)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

"""Classify proposal frame sequences with Qwen3-VL next-token scores.

Each proposal is passed as a configurable number of ordered images.  The model predicts one of ten
digit tokens; their logits are renormalized into a closed-set class
distribution.  This avoids contact-sheet cropping and self-reported confidence.
"""

import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path

from tools.run_clip_target_adapter import ACTION_PHRASES, LABELS
from tools.run_clip_then_llava import frame_paths, read_jsonl, write_json


LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
DIGITS = tuple(str(index) for index in range(len(LABELS)))
PROMPT_VERSION = "qwen3-vl-8b-ordered-images-digit-logits-v1"
WIRE_API = "qwen3-vl-next-token-classification"
PROMPT_SHA256 = hashlib.sha256(
    json.dumps(
        {
            "version": PROMPT_VERSION,
            "labels": LABELS,
            "phrases": ACTION_PHRASES,
        },
        sort_keys=True,
    ).encode("utf-8")
).hexdigest()


def digit_to_label(candidate_id):
    """Deterministically shuffle label positions to cancel digit-position bias."""
    seed = int.from_bytes(
        hashlib.sha256(str(candidate_id).encode("utf-8")).digest()[:8], "big"
    )
    labels = list(LABELS)
    random.Random(seed).shuffle(labels)
    return tuple(labels)


def build_prompt(candidate_id, frame_count=8):
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    mapping = digit_to_label(candidate_id)
    definitions = {label: phrase for label, phrase in zip(LABELS, ACTION_PHRASES)}
    options = [
        "{}: {} ({})".format(digit, label, definitions[label])
        for digit, label in zip(DIGITS, mapping)
    ]
    intro = (
        "The {} images above are frames sampled uniformly from one temporal proposal "
        "in chronological order. Classify the single dominant hand-object action. "
        "Choose exactly one option below.\n"
    ).format(frame_count)
    return (
        intro
        + "\n".join(options)
        + "\nAnswer immediately with exactly one digit from 0 to 9. Do not add "
        "words, punctuation, explanation, or formatting."
    )


def prompt_sha256(candidate_id, frame_count=8):
    return hashlib.sha256(
        build_prompt(candidate_id, frame_count).encode("utf-8")
    ).hexdigest()


def build_messages(paths, prompt, min_pixels, max_pixels):
    content = [{"type": "text", "text": "Chronological proposal frames:"}]
    for index, path in enumerate(paths, 1):
        content.append({"type": "text", "text": "FRAME {}".format(index)})
        content.append({
            "type": "image",
            "image": Path(path).resolve().as_uri(),
            "min_pixels": int(min_pixels),
            "max_pixels": int(max_pixels),
        })
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def resolve_digit_token_ids(tokenizer):
    token_ids = []
    for digit in DIGITS:
        encoded = tokenizer.encode(digit, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(
                "digit {!r} is not one tokenizer token: {}".format(digit, encoded)
            )
        token_ids.append(int(encoded[0]))
    if len(set(token_ids)) != len(token_ids):
        raise ValueError("digit token IDs are not unique: {}".format(token_ids))
    return tuple(token_ids)


def normalize_digit_logits(digit_logits, mapping, temperature=1.0):
    if temperature <= 0:
        raise ValueError("score temperature must be positive")
    if len(digit_logits) != len(DIGITS) or len(mapping) != len(DIGITS):
        raise ValueError("expected ten digit logits and ten mapped labels")
    scaled = [float(value) / temperature for value in digit_logits]
    maximum = max(scaled)
    weights = [math.exp(value - maximum) for value in scaled]
    total = sum(weights)
    digit_probabilities = [value / total for value in weights]
    class_probabilities = [0.0] * len(LABELS)
    class_logits = [0.0] * len(LABELS)
    for digit_index, label in enumerate(mapping):
        class_id = LABEL_TO_ID[label]
        class_probabilities[class_id] = digit_probabilities[digit_index]
        class_logits[class_id] = float(digit_logits[digit_index])
    order = sorted(range(len(LABELS)), key=class_probabilities.__getitem__, reverse=True)
    top1, top2 = order[:2]
    entropy = -sum(
        probability * math.log(max(probability, 1e-12))
        for probability in class_probabilities
    )
    return {
        "prediction": LABELS[top1],
        "class_id": top1,
        "confidence": class_probabilities[top1],
        "margin": class_probabilities[top1] - class_probabilities[top2],
        "entropy": entropy,
        "class_probabilities": class_probabilities,
        "class_logits": class_logits,
        "digit_probabilities": digit_probabilities,
    }


def load_qwen3(args):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    dtype = "auto" if args.dtype == "auto" else getattr(torch, args.dtype)
    model_kwargs = {
        "dtype": dtype,
        "attn_implementation": args.attn_implementation,
        "local_files_only": args.offline,
    }
    if args.device_map != "none":
        model_kwargs["device_map"] = args.device_map
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, **model_kwargs
    )
    if args.device_map == "none":
        model.to("cuda")
    model.eval()
    processor = AutoProcessor.from_pretrained(
        args.model_path, local_files_only=args.offline
    )
    return {
        "torch": torch,
        "model": model,
        "processor": processor,
        "process_vision_info": process_vision_info,
        "digit_token_ids": resolve_digit_token_ids(processor.tokenizer),
        "device": model.device,
        "model_revision": getattr(model.config, "_commit_hash", None),
    }


def infer_one(runtime, paths, candidate_id, args):
    torch = runtime["torch"]
    prompt = build_prompt(candidate_id, args.frame_count)
    mapping = digit_to_label(candidate_id)
    messages = build_messages(paths, prompt, args.min_pixels, args.max_pixels)
    chat_template_kwargs = runtime.get("chat_template_kwargs", {})
    text = runtime["processor"].apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **chat_template_kwargs
    )
    image_inputs, video_inputs = runtime["process_vision_info"](
        messages, image_patch_size=16
    )
    inputs = runtime["processor"](
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(runtime["device"])
    with torch.inference_mode():
        output = runtime["model"](
            **inputs, use_cache=False, logits_to_keep=1, return_dict=True
        )
    next_logits = output.logits[0, -1].float()
    token_ids = torch.tensor(
        runtime["digit_token_ids"], device=next_logits.device, dtype=torch.long
    )
    digit_logits_tensor = next_logits.index_select(0, token_ids)
    candidate_log_mass = (
        torch.logsumexp(digit_logits_tensor, dim=0)
        - torch.logsumexp(next_logits, dim=0)
    )
    result = normalize_digit_logits(
        digit_logits_tensor.detach().cpu().tolist(),
        mapping,
        args.score_temperature,
    )
    result.update({
        "candidate_mass": float(candidate_log_mass.exp().clamp(0, 1).cpu()),
        "digit_to_label": list(mapping),
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
    })
    return result


def prediction_is_current(row, args):
    candidate_id = row.get("candidate_id")
    frame_count = getattr(args, "frame_count", 8)
    return bool(
        candidate_id
        and row.get("status") == "ok"
        and row.get("model") == args.model_path
        and row.get("prompt_version") == PROMPT_VERSION
        and row.get("prompt_sha256") == prompt_sha256(
            candidate_id, frame_count
        )
        and row.get("wire_api") == WIRE_API
    )


def score_manifest(runtime, manifest_path, output_path, frame_cache,
                   cache_namespace, args):
    manifest = read_jsonl(manifest_path)
    if not manifest:
        raise ValueError("manifest is empty: {}".format(manifest_path))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = {
        row["candidate_id"]
        for row in read_jsonl(output_path)
        if prediction_is_current(row, args)
    }
    pending = [row for row in manifest if row["candidate_id"] not in completed]
    if args.limit > 0:
        pending = pending[:args.limit]
    for number, row in enumerate(pending, 1):
        try:
            paths = frame_paths(
                frame_cache,
                cache_namespace,
                row["candidate_id"],
                frame_count=args.frame_count,
            )
            scores = infer_one(runtime, paths, row["candidate_id"], args)
            result = {
                **row,
                "status": "ok",
                "model": args.model_path,
                "model_revision": runtime["model_revision"],
                "wire_api": WIRE_API,
                "prompt_version": PROMPT_VERSION,
                "base_prompt_sha256": PROMPT_SHA256,
                "frame_count": args.frame_count,
                "frame_paths": [str(path) for path in paths],
                "min_pixels": args.min_pixels,
                "max_pixels": args.max_pixels,
                "score_temperature": args.score_temperature,
                **scores,
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
    return read_jsonl(output_path)


def _mean(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def source_gate(manifest_path, predictions_path, output_path, args):
    from tools.score_vlm_target_adapter import summarize

    manifest = read_jsonl(manifest_path)
    predictions = {
        row["candidate_id"]: row
        for row in read_jsonl(predictions_path)
        if prediction_is_current(row, args)
    }
    rows = []
    for row in manifest:
        prediction = predictions.get(row["candidate_id"])
        if prediction is None:
            continue
        predicted_label = (
            prediction["prediction"]
            if float(prediction["confidence"]) >= args.min_confidence
            else "ABSTAIN"
        )
        rows.append({
            **row,
            "prediction": predicted_label,
            "vlm_confidence": float(prediction["confidence"]),
            "candidate_mass": float(prediction["candidate_mass"]),
            "margin": float(prediction["margin"]),
        })
    result = summarize(rows, len(manifest), {
        "min_success_rate": args.min_success_rate,
        "min_consensus_coverage": args.min_consensus_coverage,
        "min_precision_gain": args.min_precision_gain,
        "min_background_precision": args.min_background_precision,
    })

    known_rows = [row for row in rows if row.get("gt_assigned_class_ids")]
    known_correct = sum(
        LABEL_TO_ID.get(row["prediction"], -1) in row["gt_assigned_class_ids"]
        for row in known_rows
    )
    per_class = {}
    recalls = []
    for class_id, label in enumerate(LABELS):
        class_rows = [
            row for row in known_rows
            if class_id in row["gt_assigned_class_ids"]
        ]
        correct = sum(row["prediction"] == label for row in class_rows)
        recall = correct / len(class_rows) if class_rows else None
        if recall is not None:
            recalls.append(recall)
        per_class[label] = {
            "support": len(class_rows),
            "correct": correct,
            "recall": recall,
        }

    prediction_counts = result["metrics"]["prediction_counts"]
    largest_count = max(prediction_counts.values(), default=0)
    largest_share = largest_count / len(rows) if rows else None
    mean_candidate_mass = _mean(rows, "candidate_mass")
    constraint_pass = bool(
        mean_candidate_mass is not None
        and mean_candidate_mass >= args.min_candidate_mass
    )
    distribution_pass = bool(
        largest_share is not None
        and largest_share <= args.max_prediction_share
    )
    result["adapter_pass_before_qwen_checks"] = result["adapter_pass"]
    result["adapter_pass"] = bool(
        result["adapter_pass"] and constraint_pass and distribution_pass
    )
    result["qwen_checks"] = {
        "constraint_pass": constraint_pass,
        "distribution_pass": distribution_pass,
        "min_mean_candidate_mass": args.min_candidate_mass,
        "max_prediction_share_threshold": args.max_prediction_share,
        "mean_candidate_mass": mean_candidate_mass,
        "largest_prediction_share": largest_share,
    }
    result["classification_metrics"] = {
        "known_action_count": len(known_rows),
        "known_action_accuracy": (
            known_correct / len(known_rows) if known_rows else None
        ),
        "macro_recall": sum(recalls) / len(recalls) if recalls else None,
        "per_class": per_class,
        "mean_confidence": _mean(rows, "vlm_confidence"),
        "mean_margin": _mean(rows, "margin"),
    }
    result["vlm_model"] = args.model_path
    result["prompt_version"] = PROMPT_VERSION
    result["base_prompt_sha256"] = PROMPT_SHA256
    result["vlm_confidence_threshold"] = args.min_confidence
    write_json(output_path, result)
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--target-manifest", required=True)
    parser.add_argument("--source-frame-cache", required=True)
    parser.add_argument("--target-frame-cache", required=True)
    parser.add_argument("--source-cache-namespace", default="s5_source")
    parser.add_argument("--target-cache-namespace", default="s5")
    parser.add_argument("--source-output", required=True)
    parser.add_argument("--target-output", required=True)
    parser.add_argument("--source-gate-output", required=True)
    parser.add_argument("--model-path", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--device-map", choices=("none", "auto"), default="none")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument(
        "--dtype", choices=("auto", "bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument("--frame-count", type=int, default=8)
    parser.add_argument("--min-pixels", type=int, default=224 * 224)
    parser.add_argument("--max-pixels", type=int, default=448 * 448)
    parser.add_argument("--score-temperature", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force-target", action="store_true")
    parser.add_argument("--min-success-rate", type=float, default=0.95)
    parser.add_argument("--min-consensus-coverage", type=float, default=0.20)
    parser.add_argument("--min-precision-gain", type=float, default=0.03)
    parser.add_argument("--min-background-precision", type=float, default=0.80)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--min-candidate-mass", type=float, default=0.50)
    parser.add_argument("--max-prediction-share", type=float, default=0.60)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.frame_count <= 0:
        raise ValueError("frame count must be positive")
    if args.min_pixels <= 0 or args.max_pixels < args.min_pixels:
        raise ValueError("pixel limits must satisfy 0 < min <= max")
    if args.score_temperature <= 0:
        raise ValueError("score temperature must be positive")
    runtime = load_qwen3(args)
    score_manifest(
        runtime,
        args.source_manifest,
        args.source_output,
        args.source_frame_cache,
        args.source_cache_namespace,
        args,
    )
    gate = source_gate(
        args.source_manifest,
        args.source_output,
        args.source_gate_output,
        args,
    )
    print(json.dumps(gate, indent=2, sort_keys=True), flush=True)
    if not gate["adapter_pass"] and not args.force_target:
        print("source gate 未通过，跳过 target；如需强制运行请加 --force-target", flush=True)
        return 0
    score_manifest(
        runtime,
        args.target_manifest,
        args.target_output,
        args.target_frame_cache,
        args.target_cache_namespace,
        args,
    )
    return 0


if __name__ == "__main__":
    main()

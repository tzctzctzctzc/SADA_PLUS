"""Ground a classified action to one of eight cached proposal frames."""

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

from tools.run_clip_target_adapter import ACTION_PHRASES, LABELS
from tools.run_clip_then_llava import frame_paths, read_jsonl
from tools.run_qwen3_vl_classification import build_messages, load_qwen3


PROMPT_VERSION = "qwen3-vl-ordered-images-frame-grounding-v1"
WIRE_API = "qwen3-vl-next-token-frame-classification"
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
LABEL_TO_PHRASE = dict(zip(LABELS, ACTION_PHRASES))


def digit_to_frame(candidate_id, frame_count=8):
    if not 1 <= frame_count <= 10:
        raise ValueError("frame_count must be between 1 and 10")
    seed = int.from_bytes(
        hashlib.sha256(
            (str(candidate_id) + ":frame-grounding").encode("utf-8")
        ).digest()[:8],
        "big",
    )
    frames = list(range(frame_count))
    random.Random(seed).shuffle(frames)
    return tuple(frames)


def build_prompt(candidate_id, action_label, frame_count=8):
    mapping = digit_to_frame(candidate_id, frame_count)
    options = [
        "{}: FRAME {}".format(digit, frame_index + 1)
        for digit, frame_index in enumerate(mapping)
    ]
    return (
        "The images above are chronological frames sampled uniformly from one "
        "temporal proposal. The dominant action is {} ({}). Select the single "
        "frame that most clearly shows the hand-object interaction for this "
        "action. If several frames show it, select the moment of strongest "
        "interaction.\n".format(action_label, LABEL_TO_PHRASE[action_label])
        + "\n".join(options)
        + "\nAnswer immediately with exactly one digit from 0 to {}. Do not add "
        "words, punctuation, explanation, or formatting.".format(frame_count - 1)
    )


def normalize_frame_logits(logits, mapping, temperature=1.0):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = [float(value) / temperature for value in logits]
    maximum = max(scaled)
    weights = [math.exp(value - maximum) for value in scaled]
    total = sum(weights)
    digit_probabilities = [value / total for value in weights]
    frame_probabilities = [0.0] * len(mapping)
    for digit, frame_index in enumerate(mapping):
        frame_probabilities[frame_index] = digit_probabilities[digit]
    peak = max(range(len(mapping)), key=frame_probabilities.__getitem__)
    ordered = sorted(frame_probabilities, reverse=True)
    return {
        "peak_frame_index": peak,
        "frame_probabilities": frame_probabilities,
        "frame_confidence": ordered[0],
        "frame_margin": ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0],
        "digit_to_frame": list(mapping),
    }


def infer_one(runtime, paths, row, args):
    torch = runtime["torch"]
    prompt = build_prompt(row["candidate_id"], row["prediction"], args.frame_count)
    messages = build_messages(paths, prompt, args.min_pixels, args.max_pixels)
    text = runtime["processor"].apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **runtime.get("chat_template_kwargs", {}),
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
        runtime["digit_token_ids"][:args.frame_count],
        device=next_logits.device,
        dtype=torch.long,
    )
    candidate_logits = next_logits.index_select(0, token_ids)
    candidate_mass = (
        torch.logsumexp(candidate_logits, dim=0)
        - torch.logsumexp(next_logits, dim=0)
    ).exp().clamp(0, 1)
    result = normalize_frame_logits(
        candidate_logits.detach().cpu().tolist(),
        digit_to_frame(row["candidate_id"], args.frame_count),
        args.score_temperature,
    )
    result.update({
        "frame_candidate_mass": float(candidate_mass.cpu()),
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
    })
    return result


def score_predictions(runtime, args):
    rows = [
        row for row in read_jsonl(args.predictions)
        if row.get("status") == "ok"
        and row.get("prediction") in LABEL_TO_ID
        and float(row.get("confidence", 0.0)) >= args.min_class_confidence
    ]
    completed = {
        row["candidate_id"]
        for row in read_jsonl(args.output)
        if row.get("status") == "ok"
        and row.get("prompt_version") == PROMPT_VERSION
    }
    pending = [row for row in rows if row["candidate_id"] not in completed]
    if args.limit > 0:
        pending = pending[:args.limit]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for number, row in enumerate(pending, 1):
        try:
            paths = frame_paths(
                args.frame_cache,
                args.cache_namespace,
                row["candidate_id"],
                args.frame_count,
            )
            grounding = infer_one(runtime, paths, row, args)
            result = {
                **row,
                **grounding,
                "status": "ok",
                "grounding_model": args.model_path,
                "grounding_model_revision": runtime["model_revision"],
                "prompt_version": PROMPT_VERSION,
                "wire_api": WIRE_API,
                "frame_count": args.frame_count,
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
    return [row for row in read_jsonl(output) if row.get("status") == "ok"]


def frame_timestamp(row, frame_index):
    start = float(row["proposal_start_sec"])
    end = float(row["proposal_end_sec"])
    return start + (end - start) * (float(frame_index) + 0.5) / int(
        row["frame_count"]
    )


def grounding_audit(rows, database):
    def hits(row, frame_index):
        timestamp = frame_timestamp(row, frame_index)
        class_id = LABEL_TO_ID[row["prediction"]]
        return any(
            int(annotation["label_id"]) == class_id
            and float(annotation["segment"][0]) <= timestamp
            <= float(annotation["segment"][1])
            for annotation in database[row["video_id"]]["annotations"]
        )

    eligible = [row for row in rows if row["video_id"] in database]
    midpoint_index = lambda row: (int(row["frame_count"]) - 1) // 2
    peak_hits = sum(hits(row, int(row["peak_frame_index"])) for row in eligible)
    midpoint_hits = sum(hits(row, midpoint_index(row)) for row in eligible)
    oracle_hits = sum(
        any(hits(row, index) for index in range(int(row["frame_count"])))
        for row in eligible
    )
    count = len(eligible)
    return {
        "count": count,
        "peak_hit_rate": peak_hits / count if count else None,
        "midpoint_hit_rate": midpoint_hits / count if count else None,
        "peak_gain_vs_midpoint": (
            (peak_hits - midpoint_hits) / count if count else None
        ),
        "oracle_frame_hit_rate": oracle_hits / count if count else None,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--frame-cache", required=True)
    parser.add_argument("--cache-namespace", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--device-map", choices=("none", "auto"), default="auto")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument(
        "--dtype", choices=("auto", "bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument("--frame-count", type=int, default=8)
    parser.add_argument("--min-pixels", type=int, default=224 * 224)
    parser.add_argument("--max-pixels", type=int, default=448 * 448)
    parser.add_argument("--score-temperature", type=float, default=1.0)
    parser.add_argument("--min-class-confidence", type=float, default=0.7)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 1 <= args.frame_count <= 10:
        raise ValueError("frame_count must be between 1 and 10")
    runtime = load_qwen3(args)
    rows = score_predictions(runtime, args)
    if args.annotations:
        database = json.loads(
            Path(args.annotations).read_text(encoding="utf-8")
        )["database"]
        print(json.dumps(grounding_audit(rows, database), indent=2))


if __name__ == "__main__":
    main()

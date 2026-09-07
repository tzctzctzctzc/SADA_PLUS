"""Classify proposal frames with a vision-capable Responses API model."""

import argparse
import base64
import hashlib
import io
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tools.run_clip_target_adapter import ACTION_PHRASES, LABELS
from tools.run_clip_then_llava import frame_paths, read_jsonl
from tools.run_qwen3_vl_classification import digit_to_label, LABEL_TO_ID


PROMPT_VERSION = "responses-api-ordered-images-self-reported-probabilities-v2"
PROMPT_SHA256 = hashlib.sha256(
    json.dumps(
        {"version": PROMPT_VERSION, "labels": LABELS, "phrases": ACTION_PHRASES},
        sort_keys=True,
    ).encode("utf-8")
).hexdigest()
WIRE_API = "openai-responses-vision-self-reported-probabilities"


def image_data_url(path, max_pixels):
    from PIL import Image

    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.width * image.height > max_pixels:
            scale = math.sqrt(max_pixels / (image.width * image.height))
            size = (
                max(1, round(image.width * scale)),
                max(1, round(image.height * scale)),
            )
            image = image.resize(size, Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return "data:image/jpeg;base64," + encoded


def build_prompt(candidate_id):
    mapping = digit_to_label(candidate_id)
    definitions = {label: phrase for label, phrase in zip(LABELS, ACTION_PHRASES)}
    options = [
        "{}: {} ({})".format(digit, label, definitions[label])
        for digit, label in zip(range(len(mapping)), mapping)
    ]
    return (
        "The eight images are frames sampled uniformly from one temporal proposal "
        "in chronological order. Classify the single dominant hand-object action.\n"
        + "\n".join(options)
        + "\nReturn only a JSON object with exactly these keys: digit and probabilities. "
        "digit must be an integer from 0 to 9. probabilities must be ten numeric "
        "values in digit order 0 through 9, each at least 0, summing to 1. The "
        "digit must equal the index of the largest probability. Do not add markdown "
        "or explanation."
    )


def build_input(paths, prompt, max_pixels):
    content = [{"type": "input_text", "text": "Chronological proposal frames:"}]
    for index, path in enumerate(paths, 1):
        content.append({"type": "input_text", "text": "FRAME {}".format(index)})
        content.append(
            {
                "type": "input_image",
                "image_url": image_data_url(path, max_pixels),
            }
        )
    content.append({"type": "input_text", "text": prompt})
    return [{"role": "user", "content": content}]


def response_text(body):
    direct = body.get("output_text")
    if direct:
        return str(direct)
    texts = []
    for item in body.get("output") or []:
        for content in item.get("content") or []:
            if content.get("type") == "output_text" and content.get("text"):
                texts.append(str(content["text"]))
    if not texts:
        raise ValueError("Responses API returned no output text")
    return "\n".join(texts)


def parse_json_output(text):
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def parse_api_response(body, candidate_id):
    raw = response_text(body)
    parsed = parse_json_output(raw)
    probabilities = parsed.get("probabilities")
    if not isinstance(probabilities, list) or len(probabilities) != len(LABELS):
        raise ValueError("API response must contain ten probabilities")
    probabilities = [float(value) for value in probabilities]
    if any(not math.isfinite(value) or value < 0 for value in probabilities):
        raise ValueError("API probabilities must be finite and non-negative")
    total = sum(probabilities)
    if total <= 0:
        raise ValueError("API probabilities have zero mass")
    probabilities = [value / total for value in probabilities]
    digit = int(parsed.get("digit"))
    top_digit = max(range(len(probabilities)), key=probabilities.__getitem__)
    if digit != top_digit:
        raise ValueError("API digit disagrees with probability argmax")

    mapping = digit_to_label(candidate_id)
    class_probabilities = [0.0] * len(LABELS)
    for digit_index, label in enumerate(mapping):
        class_probabilities[LABEL_TO_ID[label]] = probabilities[digit_index]
    class_id = max(
        range(len(class_probabilities)), key=class_probabilities.__getitem__
    )
    ordered = sorted(class_probabilities, reverse=True)
    return {
        "prediction": LABELS[class_id],
        "class_id": class_id,
        "confidence": ordered[0],
        "margin": ordered[0] - ordered[1],
        "confidence_available": True,
        "confidence_semantics": "model_self_reported_probability_not_token_logits",
        "class_probabilities": class_probabilities,
        "class_logits": [math.log(max(value, 1e-12)) for value in class_probabilities],
        "digit": digit,
        "digit_to_label": list(mapping),
        "raw_content": raw[:1000],
        "usage": body.get("usage"),
        "api_model": body.get("model"),
    }


def request_one(args, api_key, input_items, session_id):
    payload = {
        "model": args.model,
        "input": input_items,
        "max_output_tokens": args.max_output_tokens,
    }
    request = urllib.request.Request(
        args.endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
            "User-Agent": args.user_agent,
            "x-opencode-session": session_id,
        },
        method="POST",
    )
    last_error = None
    for attempt in range(args.max_retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
            last_error = error
            if attempt >= args.max_retries:
                break
            time.sleep(min(60, 2 ** attempt))
    raise RuntimeError("API request failed after retries: {}".format(last_error))


def prediction_is_current(row, args):
    candidate_id = row.get("candidate_id")
    return bool(
        candidate_id
        and row.get("status") == "ok"
        and row.get("model") == args.model
        and row.get("prompt_version") == PROMPT_VERSION
        and row.get("max_pixels") == args.max_pixels
        and row.get("prompt_sha256")
        == hashlib.sha256(build_prompt(candidate_id).encode("utf-8")).hexdigest()
        and row.get("wire_api") == WIRE_API
    )


def score_manifest(manifest_path, output_path, frame_cache, namespace, args, api_key):
    manifest = read_jsonl(manifest_path)
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
    def score_row(row):
        try:
            paths = frame_paths(
                frame_cache, namespace, row["candidate_id"], args.frame_count
            )
            prompt = build_prompt(row["candidate_id"])
            parsed = None
            for response_attempt in range(args.response_retries + 1):
                body = request_one(
                    args,
                    api_key,
                    build_input(paths, prompt, args.max_pixels),
                    "{}-{}".format(args.session_id, row["candidate_id"]),
                )
                try:
                    parsed = parse_api_response(body, row["candidate_id"])
                    break
                except ValueError:
                    if response_attempt >= args.response_retries:
                        raise
            assert parsed is not None
            result = {
                **row,
                "status": "ok",
                "model": args.model,
                "wire_api": WIRE_API,
                "prompt_version": PROMPT_VERSION,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "base_prompt_sha256": PROMPT_SHA256,
                "frame_count": args.frame_count,
                "frame_paths": [str(path) for path in paths],
                "max_pixels": args.max_pixels,
                **parsed,
            }
        except Exception as error:
            result = {
                "candidate_id": row["candidate_id"],
                "video_id": row.get("video_id"),
                "status": "error",
                "error": "{}: {}".format(type(error).__name__, error),
            }
        return result

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(score_row, row) for row in pending]
        for number, future in enumerate(as_completed(futures), 1):
            result = future.result()
            with output_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(result, sort_keys=True) + "\n")
            print(
                "{}/{} {} {}".format(
                    number,
                    len(pending),
                    result["candidate_id"],
                    result["status"],
                ),
                flush=True,
            )


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
        if prediction is not None:
            rows.append(
                {
                    **row,
                    "prediction": prediction["prediction"],
                    "vlm_confidence": prediction["confidence"],
                }
            )
    result = summarize(
        rows,
        len(manifest),
        {
            "min_success_rate": args.min_success_rate,
            "min_consensus_coverage": args.min_consensus_coverage,
            "min_precision_gain": args.min_precision_gain,
            "min_background_precision": args.min_background_precision,
        },
    )
    result.update(
        {
            "vlm_model": args.model,
            "prompt_version": PROMPT_VERSION,
            "base_prompt_sha256": PROMPT_SHA256,
            "confidence_available": True,
            "confidence_semantics": "model_self_reported_probability_not_token_logits",
            "vlm_confidence_threshold": 0.0,
        }
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
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
    parser.add_argument(
        "--endpoint", default="https://opencode.ai/zen/go/v1/responses"
    )
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--api-key-env", default="OPENCODE_API_KEY")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--user-agent", default="sada-benchmark/1.0")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--max-pixels", type=int, default=448 * 448)
    parser.add_argument("--response-retries", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--frame-count", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--force-target", action="store_true")
    parser.add_argument("--min-success-rate", type=float, default=0.95)
    parser.add_argument("--min-consensus-coverage", type=float, default=0.20)
    parser.add_argument("--min-precision-gain", type=float, default=0.03)
    parser.add_argument("--min-background-precision", type=float, default=0.80)
    return parser.parse_args()


def main():
    args = parse_args()
    if (
        args.frame_count <= 0
        or args.workers <= 0
        or args.max_pixels <= 0
        or args.response_retries < 0
        or not args.session_id.strip()
    ):
        raise ValueError("frame count, workers, pixels, retries, and session must be valid")
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError("missing API key environment variable")
    score_manifest(
        args.source_manifest,
        args.source_output,
        args.source_frame_cache,
        args.source_cache_namespace,
        args,
        api_key,
    )
    gate = source_gate(
        args.source_manifest, args.source_output, args.source_gate_output, args
    )
    print(json.dumps(gate, indent=2, sort_keys=True), flush=True)
    if not args.source_only and (gate["adapter_pass"] or args.force_target):
        score_manifest(
            args.target_manifest,
            args.target_output,
            args.target_frame_cache,
            args.target_cache_namespace,
            args,
            api_key,
        )


if __name__ == "__main__":
    main()

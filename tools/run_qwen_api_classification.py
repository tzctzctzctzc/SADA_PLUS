"""Run the S5 proposal classifier through an OpenAI-compatible vision API."""

import argparse
import base64
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from tools.run_clip_target_adapter import LABELS
from tools.run_clip_then_llava import frame_paths, read_jsonl
from tools.run_qwen3_vl_classification import (
    build_prompt,
    digit_to_label,
    LABEL_TO_ID,
)


PROMPT_VERSION = "qwen3-api-ordered-images-digit-v1"
PROMPT_SHA256 = hashlib.sha256(
    (PROMPT_VERSION + "|" + "|".join(LABELS)).encode("utf-8")
).hexdigest()
WIRE_API = "openai-compatible-vision-chat-completions"


def image_data_url(path):
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return "data:image/jpeg;base64," + encoded


def build_api_content(paths, prompt):
    content = []
    for index, path in enumerate(paths, 1):
        content.append({"type": "text", "text": "FRAME {}".format(index)})
        content.append({
            "type": "image_url",
            "image_url": {"url": image_data_url(path)},
        })
    content.append({"type": "text", "text": prompt})
    return content


def parse_digit_output(content):
    text = (content or "").strip()
    if re.fullmatch(r"[0-9]", text):
        return int(text)
    matches = re.findall(r"(?<![0-9])([0-9])(?![0-9])", text)
    if not matches:
        raise ValueError("API response contains no class digit: {}".format(text[:300]))
    return int(matches[-1])


def parse_api_response(body, candidate_id):
    choices = body.get("choices") or []
    if not choices:
        raise ValueError("API response has no choices")
    message = choices[0].get("message") or {}
    digit = parse_digit_output(message.get("content", ""))
    mapping = digit_to_label(candidate_id)
    if digit >= len(mapping):
        raise ValueError("class digit out of range: {}".format(digit))
    label = mapping[digit]
    return {
        "prediction": label,
        "class_id": LABEL_TO_ID[label],
        "confidence": 0.0,
        "confidence_available": False,
        "confidence_semantics": "not_returned_by_api",
        "digit": digit,
        "digit_to_label": list(mapping),
        "raw_content": str(message.get("content", ""))[:500],
        "reasoning_content": str(message.get("reasoning_content", ""))[:1000],
        "usage": body.get("usage"),
        "api_model": body.get("model"),
    }


def request_one(endpoint, api_key, model, content, timeout, max_retries):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 16,
        "temperature": 0,
        "enable_thinking": False,
    }
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
            last_error = error
            if attempt >= max_retries:
                break
            time.sleep(min(60, 2 ** attempt))
    raise RuntimeError("API request failed after retries: {}".format(last_error))


def prediction_is_current(row, args):
    candidate_id = row.get("candidate_id")
    expected_prompt = hashlib.sha256(
        build_prompt(candidate_id).encode("utf-8")
    ).hexdigest() if candidate_id else None
    return bool(
        candidate_id
        and row.get("status") == "ok"
        and row.get("model") == args.model
        and row.get("prompt_version") == PROMPT_VERSION
        and row.get("prompt_sha256") == expected_prompt
        and row.get("wire_api") == WIRE_API
    )


def score_manifest(manifest_path, output_path, frame_cache, cache_namespace,
                   args, api_key):
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
                frame_cache, cache_namespace, row["candidate_id"], args.frame_count
            )
            prompt = build_prompt(row["candidate_id"])
            body = request_one(
                args.endpoint,
                api_key,
                args.model,
                build_api_content(paths, prompt),
                args.timeout,
                args.max_retries,
            )
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
                **parse_api_response(body, row["candidate_id"]),
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
        rows.append({
            **row,
            "prediction": prediction["prediction"],
            "vlm_confidence": 0.0,
        })
    result = summarize(rows, len(manifest), {
        "min_success_rate": args.min_success_rate,
        "min_consensus_coverage": args.min_consensus_coverage,
        "min_precision_gain": args.min_precision_gain,
        "min_background_precision": args.min_background_precision,
    })
    result["vlm_model"] = args.model
    result["prompt_version"] = PROMPT_VERSION
    result["base_prompt_sha256"] = PROMPT_SHA256
    result["confidence_available"] = False
    result["vlm_confidence_threshold"] = 0.0
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
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
        "--endpoint",
        default="https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    )
    parser.add_argument("--model", default="qwen3.8-max")
    parser.add_argument("--api-key-env", default="QWEN_API_KEY")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--frame-count", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force-target", action="store_true")
    parser.add_argument("--min-success-rate", type=float, default=0.95)
    parser.add_argument("--min-consensus-coverage", type=float, default=0.20)
    parser.add_argument("--min-precision-gain", type=float, default=0.03)
    parser.add_argument("--min-background-precision", type=float, default=0.80)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.frame_count <= 0:
        raise ValueError("frame count must be positive")
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError("missing API key environment variable: {}".format(args.api_key_env))
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
    if not gate["adapter_pass"] and not args.force_target:
        print("source gate 未通过，跳过 target；如需强制运行请加 --force-target", flush=True)
        return 0
    score_manifest(
        args.target_manifest,
        args.target_output,
        args.target_frame_cache,
        args.target_cache_namespace,
        args,
        api_key,
    )
    return 0


if __name__ == "__main__":
    main()

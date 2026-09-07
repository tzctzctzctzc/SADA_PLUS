"""Run LLaVA-v1.5-7B after the CLIP proposal scorer finishes.

The script deliberately keeps LLaVA outputs separate from CLIP outputs.  It
reuses the eight JPEGs already extracted by ``run_clip_target_adapter.py`` and
packs them into one numbered contact sheet because LLaVA-v1.5 is an image model
rather than a native video model.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path


LABELS = (
    "TAKE", "PUT", "WASH", "OPEN", "CLOSE", "INSERT", "TURN_ON",
    "CUT", "TURN_OFF", "POUR",
)
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
PROMPT_VERSION = "llava-v1.5-7b-contact-sheet-class-v1"
PROMPT_SHA256 = hashlib.sha256(
    (PROMPT_VERSION + "|" + "|".join(LABELS)).encode("utf-8")
).hexdigest()
DEFAULT_PROMPT = (
    "You are labeling one temporal action proposal from an egocentric video. "
    "The image is a contact sheet: panels are ordered from earliest to latest "
    "and are marked FRAME 1, FRAME 2, etc. Choose exactly one action from this "
    "closed set: TAKE, PUT, WASH, OPEN, CLOSE, INSERT, TURN_ON, CUT, "
    "TURN_OFF, POUR. Return only one JSON object with this schema: "
    '{"label":"TAKE","confidence":0.0,"reason":"short evidence"}. '
    "Confidence must be a number between 0 and 1. Do not return markdown."
)


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def candidate_frame_dir(frame_cache, cache_namespace, candidate_id):
    """Resolve both the current nested cache layout and a flat cache layout."""
    root = Path(frame_cache)
    candidates = []
    if cache_namespace:
        candidates.append(root / cache_namespace / candidate_id)
    candidates.extend((root / candidate_id, root / "frames" / candidate_id))
    for directory in candidates:
        if directory.is_dir():
            return directory
    return None


def frame_paths(frame_cache, cache_namespace, candidate_id, frame_count=8):
    directory = candidate_frame_dir(frame_cache, cache_namespace, candidate_id)
    if directory is None:
        raise FileNotFoundError(
            "frame cache not found for {} under {}".format(
                candidate_id, frame_cache
            )
        )
    frames = []
    for index in range(1, frame_count + 1):
        path = directory / "frame_{:02d}.jpg".format(index)
        if not path.is_file() or not path.stat().st_size:
            raise FileNotFoundError("missing frame: {}".format(path))
        frames.append(path)
    return frames


def make_contact_sheet(paths, columns=4, tile_size=(320, 240), label_height=28):
    """Create a numbered RGB PIL image without writing an extra cache file."""
    from PIL import Image, ImageDraw, ImageFont

    if not paths:
        raise ValueError("at least one frame is required")
    columns = max(1, int(columns))
    rows = (len(paths) + columns - 1) // columns
    width, height = tile_size
    sheet = Image.new("RGB", (columns * width, rows * (height + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for index, path in enumerate(paths):
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.thumbnail((width, height))
            tile = Image.new("RGB", (width, height), "black")
            left = (width - image.width) // 2
            top = (height - image.height) // 2
            tile.paste(image, (left, top))
        x = (index % columns) * width
        y = (index // columns) * (height + label_height)
        sheet.paste(tile, (x, y))
        draw.rectangle((x, y + height, x + width, y + height + label_height), fill="white")
        draw.text((x + 6, y + height + 6), "FRAME {}".format(index + 1), fill="black", font=font)
    return sheet


def parse_model_output(text):
    """Parse strict JSON first, then use a conservative label fallback."""
    text = (text or "").strip()
    payload = None
    for match in re.finditer(r"\{.*?\}", text, flags=re.DOTALL):
        try:
            payload = json.loads(match.group(0))
            break
        except json.JSONDecodeError:
            continue
    if isinstance(payload, dict):
        raw_label = payload.get("label", payload.get("prediction", ""))
        label = str(raw_label).strip().upper().replace(" ", "_")
        if label in LABEL_TO_ID:
            try:
                confidence = float(payload.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            return {
                "prediction": label,
                "class_id": LABEL_TO_ID[label],
                "confidence": max(0.0, min(1.0, confidence)),
                "reason": str(payload.get("reason", ""))[:500],
                "parse_mode": "json",
            }
    upper = text.upper().replace("-", "_")
    matches = [label for label in LABELS if re.search(r"\b{}\b".format(label), upper)]
    if len(matches) == 1:
        return {
            "prediction": matches[0],
            "class_id": LABEL_TO_ID[matches[0]],
            "confidence": 0.0,
            "reason": text[:500],
            "parse_mode": "label_fallback",
        }
    raise ValueError("could not parse one action label from model output: {}".format(text[:500]))


def clip_predictions_complete(paths, manifest_path):
    """Return true when all manifest IDs occur as successful CLIP rows."""
    expected = {row["candidate_id"] for row in read_jsonl(manifest_path)}
    if not expected:
        raise ValueError("manifest has no candidate rows: {}".format(manifest_path))
    observed = set()
    for path in paths:
        observed.update(
            row["candidate_id"]
            for row in read_jsonl(path)
            if row.get("status") == "ok" and "candidate_id" in row
        )
    return expected.issubset(observed)


def wait_for_clip(source_manifest, source_predictions, target_manifest,
                  target_predictions, poll_seconds, timeout_seconds):
    started = time.time()
    while True:
        source_done = clip_predictions_complete(source_predictions, source_manifest)
        target_done = clip_predictions_complete(target_predictions, target_manifest)
        print(
            "等待 CLIP：source={} target={}".format(source_done, target_done),
            flush=True,
        )
        if source_done and target_done:
            return
        if timeout_seconds > 0 and time.time() - started >= timeout_seconds:
            raise TimeoutError("CLIP outputs did not complete before timeout")
        time.sleep(poll_seconds)


def load_llava(args):
    """Load LLaVA lazily so helper functions remain testable without LLaVA."""
    repo = Path(args.llava_repo).expanduser()
    if repo.is_dir():
        sys.path.insert(0, str(repo))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates, SeparatorStyle
    from llava.mm_utils import (
        KeywordsStoppingCriteria,
        get_model_name_from_path,
        tokenizer_image_token,
    )
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    import torch

    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = args.model_name or get_model_name_from_path(model_path)
    # A Hugging Face cache snapshot ends in a hash, which hides the model name
    # expected by the LLaVA builder's dispatch logic.
    if args.model_name is None and "llava-v1.5" in model_path.lower():
        model_name = "llava-v1.5-7b"
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path,
        None,
        model_name,
        load_4bit=args.load_4bit,
        load_8bit=args.load_8bit,
        device_map="auto",
        attn_implementation=args.attn_implementation,
    )
    model.eval()
    conv_mode = args.conv_mode or ("llava_v1" if "v1" in model_name.lower() else "llava_v0")
    if conv_mode not in conv_templates:
        raise ValueError("unknown LLaVA conversation mode: {}".format(conv_mode))
    return {
        "torch": torch,
        "tokenizer": tokenizer,
        "model": model,
        "image_processor": image_processor,
        "conv_templates": conv_templates,
        "conv_mode": conv_mode,
        "separator_style": SeparatorStyle,
        "image_token_index": IMAGE_TOKEN_INDEX,
        "default_image_token": DEFAULT_IMAGE_TOKEN,
        "tokenizer_image_token": tokenizer_image_token,
        "stopping_criteria": KeywordsStoppingCriteria,
    }


def infer_one(runtime, sheet, prompt, max_new_tokens, temperature):
    torch = runtime["torch"]
    tokenizer = runtime["tokenizer"]
    model = runtime["model"]
    image_processor = runtime["image_processor"]
    conv = runtime["conv_templates"][runtime["conv_mode"]].copy()
    image_tensor = image_processor.preprocess(
        sheet, return_tensors="pt"
    )["pixel_values"].half().cuda()
    question = runtime["default_image_token"] + "\n" + prompt
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()
    input_ids = runtime["tokenizer_image_token"](
        full_prompt,
        tokenizer,
        runtime["image_token_index"],
        return_tensors="pt",
    ).unsqueeze(0).cuda()
    stop_str = (
        conv.sep
        if conv.sep_style != runtime["separator_style"].TWO
        else conv.sep2
    )
    stopping = runtime["stopping_criteria"](
        [stop_str], tokenizer, input_ids
    )
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=temperature > 0,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping],
        )
    text = tokenizer.decode(
        output_ids[0, input_ids.shape[1]:],
        skip_special_tokens=True,
    ).strip()
    return text


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
        if row.get("status") == "ok"
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
            sheet = make_contact_sheet(
                paths,
                columns=args.grid_columns,
                tile_size=(args.tile_width, args.tile_height),
            )
            raw = infer_one(
                runtime,
                sheet,
                args.prompt,
                args.max_new_tokens,
                args.temperature,
            )
            parsed = parse_model_output(raw)
            result = {
                **row,
                "status": "ok",
                "model": args.model_path,
                "wire_api": "llava-v1.5-contact-sheet",
                "prompt_version": PROMPT_VERSION,
                "prompt_sha256": PROMPT_SHA256,
                "frame_count": args.frame_count,
                "frame_paths": [str(path) for path in paths],
                "raw_output": raw,
                **parsed,
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
        if row.get("status") == "ok"
    }
    rows = []
    for row in manifest:
        prediction = predictions.get(row["candidate_id"])
        if prediction is None:
            continue
        if float(prediction.get("confidence", 0.0)) < args.min_confidence:
            predicted_label = "ABSTAIN"
        else:
            predicted_label = prediction["prediction"]
        rows.append({
            **row,
            "prediction": predicted_label,
            "vlm_confidence": float(prediction.get("confidence", 0.0)),
        })
    result = summarize(rows, len(manifest), {
        "min_success_rate": args.min_success_rate,
        "min_consensus_coverage": args.min_consensus_coverage,
        "min_precision_gain": args.min_precision_gain,
        "min_background_precision": args.min_background_precision,
    })
    result["vlm_model"] = args.model_path
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
    parser.add_argument("--model-path", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--llava-repo", required=True)
    parser.add_argument("--conv-mode", default=None)
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--frame-count", type=int, default=8)
    parser.add_argument("--grid-columns", type=int, default=4)
    parser.add_argument("--tile-width", type=int, default=320)
    parser.add_argument("--tile-height", type=int, default=240)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force-target", action="store_true")
    parser.add_argument("--wait-for-clip", action="store_true")
    parser.add_argument("--clip-source-predictions", nargs="+", default=[])
    parser.add_argument("--clip-target-predictions", nargs="+", default=[])
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--wait-timeout-seconds", type=int, default=0)
    parser.add_argument("--min-success-rate", type=float, default=0.95)
    parser.add_argument("--min-consensus-coverage", type=float, default=0.20)
    parser.add_argument("--min-precision-gain", type=float, default=0.03)
    parser.add_argument("--min-background-precision", type=float, default=0.80)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.frame_count <= 0 or args.grid_columns <= 0:
        raise ValueError("frame count and grid columns must be positive")
    if args.load_4bit and args.load_8bit:
        raise ValueError("choose at most one quantization mode")
    if args.wait_for_clip:
        if not args.clip_source_predictions or not args.clip_target_predictions:
            raise ValueError("--wait-for-clip requires both CLIP prediction lists")
        wait_for_clip(
            args.source_manifest,
            args.clip_source_predictions,
            args.target_manifest,
            args.clip_target_predictions,
            args.poll_seconds,
            args.wait_timeout_seconds,
        )

    runtime = load_llava(args)
    score_manifest(
        runtime,
        args.source_manifest,
        args.source_output,
        args.source_frame_cache,
        args.source_cache_namespace,
        args,
    )
    gate = source_gate(args.source_manifest, args.source_output, args.source_gate_output, args)
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


if __name__ == "__main__":
    main()

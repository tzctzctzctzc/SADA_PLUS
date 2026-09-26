#!/usr/bin/env python3
"""双规格 VLM 的相对破坏筛选；从不声称目标满足全部角色。

选项 D 明确表示不可观察/不确定。A/B 位置随机，输出四选项的条件
概率仅供筛选，不当作已校准的正确率。逐图复核另存记录。
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import inspect
import io
import itertools
import json
import os
import time
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

PROMPT = "Inspect only the visible '{role}' attribute of the foreground products; ignore background, brand, text and unrelated attributes. Which reference is closer to the target on that ONE attribute? Answer one letter: A = Reference A is clearly closer; B = Reference B is clearly closer; C = both are equally close or equally different; D = the attribute is not visible, ambiguous, or you are uncertain. Do not guess."


def canonical(value):
    return value.strip().lower() if isinstance(value, str) else json.dumps(value, sort_keys=True).lower()


def build_checks(rows, meta, families, limit=None, chosen_ids=None):
    """只用元数据选择可比较角色；属性取值不进入标注器输入。"""
    items, eligible = [], []
    for row in sorted(rows, key=lambda r: hashlib.sha256(r["bundle_id"].encode()).digest()):
        active = row["active_role"]
        permutations = list(itertools.permutations(range(len(row["source_ids"]))))
        negative_value = meta[row["negative_target_id"]]["attribute"][row["active_role_key"]]
        comparable = [i for i in row["verified_dod_assignments"]
                      if canonical(meta[row["source_ids"][permutations[i][active]]]["attribute"][row["active_role_key"]]) == canonical(negative_value)]
        if not comparable:
            continue
        eligible.append(row)
        correct, wrong = row["source_ids"][active], row["source_ids"][permutations[comparable[0]][active]]
        for kind in ["positive", "negative"]:
            target = row[kind + "_target_id"]
            swap = int(hashlib.sha256((row["bundle_id"] + kind).encode()).hexdigest()[:8], 16) % 2
            for mirrored in [0, 1]:
                orientation = swap ^ mirrored
                sources = [correct, wrong] if orientation == 0 else [wrong, correct]
                expected = "AB"[orientation] if kind == "positive" else "BA"[orientation]
                items.append({"bundle_id": row["bundle_id"], "kind": kind, "mirrored": mirrored,
                              "product_ids": [target] + sources, "expected": expected, "family": row["active_family"],
                              "role": row["active_role_key"], "corruption_assignment": comparable[0]})
    if limit:
        buckets = {family: [r for r in eligible if r["active_family"] == family] for family in families}
        chosen_ids = {buckets[family][i // len(buckets)]["bundle_id"]
                      for i in range(limit) for family in [list(buckets)[i % len(buckets)]]
                      if i // len(buckets) < len(buckets[family])}
        assert len(chosen_ids) == limit
    return [item for item in items if chosen_ids is None or item["bundle_id"] in chosen_ids]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["small", "large"], required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--mirror-sample", action="store_true")
    args = parser.parse_args()
    mirrored_checks = args.mirror_sample or args.limit is None
    config = json.loads(Path("configs/P051_tracebind.json").read_text())
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == config["gpu_uuid"] and torch.cuda.device_count() == 1
    uuid = str(torch.cuda.get_device_properties(0).uuid)
    assert (uuid if uuid.startswith("GPU-") else "GPU-" + uuid) == config["gpu_uuid"]
    cfg = config["data"]["visual_screening"]
    run = Path("runs/P051_tracebind")
    dataset_path = run / "orbits_byte_disjoint.jsonl"
    dataset_sha = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    rows = list(map(json.loads, dataset_path.open()))
    images = {entry["idx"]: entry for entry in json.loads((run / "orbits_image_manifest.json").read_text())["entries"]}
    meta = {row["idx"]: row for row in map(json.loads, Path(config["data"]["metadata_path"]).open())}
    if args.model == "small" and not args.limit:
        large = list(map(json.loads, (run / "visual_large.jsonl").open()))
        decisions = {}
        for result in large:
            assert result["dataset_sha256"] == dataset_sha
            decisions.setdefault(result["bundle_id"], []).append(result["pass"])
        keep = {bid for bid, values in decisions.items() if len(values) == 4 and all(values)}
        rows = [row for row in rows if row["bundle_id"] in keep]
    items = build_checks(rows, meta, config["data"]["visual_families"], args.limit)
    if not mirrored_checks:
        items = [item for item in items if item["mirrored"] == 0]
    suffix = f"_{args.limit}_{'mirror' if mirrored_checks else 'single_order'}" if args.limit else ""
    out = run / f"visual_{args.model}{suffix}.jsonl"
    done = {}
    if out.exists():
        for row in map(json.loads, out.open()):
            assert row["dataset_sha256"] == dataset_sha and row["screening_config"] == cfg
            done[(row["bundle_id"], row["kind"], row["mirrored"])] = row
    items = [item for item in items if (item["bundle_id"], item["kind"], item["mirrored"]) not in done]
    model_cfg = cfg["models"][args.model]
    model_path = Path(model_cfg["path"])
    assert json.loads((model_path / "source_manifest.json").read_text())["revision"] == model_cfg["revision"]
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, min_pixels=cfg["min_pixels"], max_pixels=cfg["max_pixels"])
    processor.tokenizer.padding_side = "left"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.bfloat16,
                                                              device_map={"": 0}, attn_implementation="sdpa").eval()
    assert "logits_to_keep" in inspect.signature(model.forward).parameters
    ids = [processor.tokenizer.encode(label, add_special_tokens=False) for label in ["A", "B", "C", "D"]]
    assert all(len(values) == 1 for values in ids)
    ids = [values[0] for values in ids]
    @functools.lru_cache(maxsize=128)
    def image(product):
        data = Path(images[product]["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == images[product]["sha256"]
        return Image.open(io.BytesIO(data)).convert("RGB")
    started = time.monotonic()
    with out.open("a") as f:
        for start in range(0, len(items), cfg["batch_size"]):
            current = items[start:start + cfg["batch_size"]]
            messages = []
            for item in current:
                content = []
                for label in ["Target product", "Reference A", "Reference B"]:
                    content += [{"type": "text", "text": label + ":\n"}, {"type": "image"}]
                content.append({"type": "text", "text": PROMPT.format(role=item["role"])})
                messages.append([{"role": "user", "content": content}])
            texts = [processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in messages]
            inputs = processor(text=texts, images=[image(product) for item in current for product in item["product_ids"]], padding=True, return_tensors="pt").to("cuda")
            assert len(inputs["image_grid_thw"]) == 3 * len(current)
            with torch.inference_mode():
                vocabulary_logits = model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1].float()
                logits = vocabulary_logits[:, ids]
                option_mass = (torch.logsumexp(logits, -1) - torch.logsumexp(vocabulary_logits, -1)).exp().cpu()
                probabilities = logits.softmax(-1).cpu()
                logits = logits.cpu()
            for item, values, probs, mass in zip(current, logits, probabilities, option_mass):
                prediction = ["A", "B", "C", "D"][int(values.argmax())]
                top = values.topk(2).values
                passed = (prediction == item["expected"] and float(probs.max()) >= cfg["minimum_option_probability"]
                          and float(mass) >= cfg["minimum_total_option_mass"]
                          and float(top[0] - top[1]) >= cfg["minimum_logit_margin"])
                result = {**item, "prediction": prediction, "pass": passed, "option_logits": values.tolist(),
                          "option_probabilities": probs.tolist(), "model": args.model, "revision": model_cfg["revision"],
                          "total_option_probability_mass": float(mass),
                          "dataset_sha256": dataset_sha, "screening_config": cfg,
                          "prompt_sha256": hashlib.sha256(PROMPT.format(role=item["role"]).encode()).hexdigest()}
                f.write(json.dumps(result) + "\n"); done[(item["bundle_id"], item["kind"], item["mirrored"])] = result
            f.flush()
            if start % (cfg["batch_size"] * 20) == 0:
                print(json.dumps({"model": args.model, "checks_complete": len(done), "remaining_new_checks": len(items) - start - len(current),
                                  "elapsed_seconds": time.monotonic() - started}), flush=True)
    results = list(done.values())
    grouped = {}
    for result in results:
        grouped.setdefault(result["bundle_id"], []).append(result)
    required = 4 if mirrored_checks else 2
    passed = {bid for bid, values in grouped.items() if len(values) == required and all(value["pass"] for value in values)}
    summary = {"model": args.model, "checks": len(results), "groups": len(grouped), "passed_groups": len(passed),
               "passed_family_counts": dict(Counter(values[0]["family"] for bid, values in grouped.items() if bid in passed)),
               "uncertain_predictions": sum(row["prediction"] == "D" for row in results),
               "scope": "relative active-role quality screening only; not absolute multi-role ground truth",
               "dataset_sha256": dataset_sha, "elapsed_seconds": time.monotonic() - started}
    (out.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

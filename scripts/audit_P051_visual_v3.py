#!/usr/bin/env python3
"""有界双视觉架构相对标注核验；阈值、提示、数据均在运行前固定。"""
from __future__ import annotations

import argparse
import functools
import hashlib
import inspect
import io
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, InternVLForConditionalGeneration, Qwen3VLForConditionalGeneration

from audit_P051_visual import PROMPT, build_checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["primary", "secondary"], required=True)
    parser.add_argument("--inspection", action="store_true")
    args = parser.parse_args()
    run = Path("runs/P051_tracebind")
    config = json.loads(Path("configs/P051_annotation_v3.json").read_text())
    training = json.loads(Path("configs/P051_tracebind.json").read_text())
    assert os.environ["CUDA_VISIBLE_DEVICES"] == training["gpu_uuid"] and torch.cuda.device_count() == 1
    uuid = str(torch.cuda.get_device_properties(0).uuid)
    assert (uuid if uuid.startswith("GPU-") else "GPU-" + uuid) == training["gpu_uuid"]
    data = Path(config["dataset"])
    assert hashlib.sha256(data.read_bytes()).hexdigest() == config["dataset_sha256"]
    rows = list(map(json.loads, data.open()))
    meta = {r["idx"]: r for r in map(json.loads, Path(training["data"]["metadata_path"]).open())}
    chosen = None
    if args.inspection:
        inspection = json.loads(Path("results/P051_tracebind/stratified_visual_inspection_raw.json").read_text())
        chosen = {r["bundle_id"] for r in inspection["groups"]}
    items = build_checks(rows, meta, training["data"]["visual_families"],
                         None if args.inspection else config["quality_pilot_groups"], chosen)
    assert len(items) == (256 if args.inspection else config["quality_pilot_groups"] * 4)
    expected_items = {(r["bundle_id"], r["kind"], r["mirrored"]): r for r in items}
    suffix = "inspection64" if args.inspection else "96_mirror"
    out = run / f"visual_v3_{args.model}_{suffix}.jsonl"
    done = {}
    if out.exists():
        for r in map(json.loads, out.open()):
            assert r["screening_config"] == config
            key = r["bundle_id"], r["kind"], r["mirrored"]
            assert key not in done and all(r[k] == v for k, v in expected_items[key].items())
            done[key] = r
    pending = [r for key, r in expected_items.items() if key not in done]
    cfg = config["models"][args.model]
    path = Path(cfg["path"])
    manifest = json.loads((path / "source_manifest.json").read_text())
    assert manifest["revision"] == cfg["revision"]
    # 下载器可继续获取另一模型，但当前模型的每个文件必须已经校验。
    assert {r["name"] for r in manifest["verified_files"]} == {r["rfilename"] for r in manifest["files"]}
    for entry in manifest["verified_files"]:
        assert (path / entry["name"]).stat().st_size == entry["bytes"]
    processor = AutoProcessor.from_pretrained(path, local_files_only=True, use_fast=config["image_processor_use_fast"])
    processor.tokenizer.padding_side = "left"
    cls = Qwen3VLForConditionalGeneration if cfg["type"] == "qwen3_vl" else InternVLForConditionalGeneration
    model = cls.from_pretrained(path, local_files_only=True, torch_dtype=torch.bfloat16,
                               device_map={"": 0}, attn_implementation=config["attention_backend"]).eval()
    assert "logits_to_keep" in inspect.signature(model.forward).parameters
    ids = [processor.tokenizer.encode(label, add_special_tokens=False) for label in "ABCD"]
    assert all(len(values) == 1 for values in ids)
    ids = [values[0] for values in ids]

    @functools.lru_cache(maxsize=128)
    def image(product):
        entry = images[product]
        payload = Path(entry["path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == entry["sha256"]
        return Image.open(io.BytesIO(payload)).convert("RGB")

    images = {e["idx"]: e for e in json.loads((run / "orbits_image_manifest.json").read_text())["entries"]}
    def prepare(current):
        messages = []
        for item in current:
            content = []
            for label in ["Target product", "Reference A", "Reference B"]:
                content += [{"type": "text", "text": label + ":\n"}, {"type": "image"}]
            content.append({"type": "text", "text": PROMPT.format(role=item["role"])})
            messages.append([{"role": "user", "content": content}])
        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                                              enable_thinking=not config["disable_thinking"]) for m in messages]
        inputs = processor(text=texts, images=[image(p) for r in current for p in r["product_ids"]],
                           padding=True, return_tensors="pt", **cfg["image_kwargs"]).to("cuda")
        token_counts = (inputs["input_ids"] == model.config.image_token_id).sum(-1)
        if cfg["type"] == "qwen3_vl":
            grid = inputs["image_grid_thw"].reshape(len(current), 3, 3)
            expected = grid.prod(-1).sum(-1) // model.config.vision_config.spatial_merge_size ** 2
            assert torch.equal(token_counts, expected)
        else:
            # 每张原图可能拆为多个 448px patch，placeholder 数必须与视觉特征数一致。
            assert int(token_counts.sum()) == len(inputs["pixel_values"]) * processor.image_seq_length
            assert (token_counts >= 3 * processor.image_seq_length).all()
        return inputs

    started = time.monotonic()
    with out.open("a") as stream, torch.inference_mode():
        for start in range(0, len(pending), cfg["batch_size"]):
            current = pending[start:start + cfg["batch_size"]]
            inputs = prepare(current)
            vocabulary = model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1].float()
            logits = vocabulary[:, ids]
            mass = (logits.logsumexp(-1) - vocabulary.logsumexp(-1)).exp().cpu()
            probabilities, logits = logits.softmax(-1).cpu(), logits.cpu()
            if start == 0:
                # 字母解码与原始概率、batch 与 single 必须来自同一计算协议。
                singles, generated, generated_first_ids = [], [], []
                for row in current:
                    single = prepare([row])
                    singles.append(model(**single, use_cache=False, logits_to_keep=1).logits[0, -1, ids].float().cpu())
                    output = model.generate(**single, do_sample=False, max_new_tokens=4, repetition_penalty=1.0)
                    generated_first_ids.append(int(output[0, single["input_ids"].shape[1]]))
                    generated.append(processor.tokenizer.decode(output[0, single["input_ids"].shape[1]:], skip_special_tokens=True).strip())
                singles = torch.stack(singles)
                checks = {"image_token_counts_match": True,
                          "single_batch_argmax_agree": bool(torch.equal(singles.argmax(-1), logits.argmax(-1))),
                          "generation_first_token_agrees_with_single_option": all(i == ids[int(v.argmax())] for i, v in zip(generated_first_ids, singles))}
                runtime = {"checks": checks, "status": "pass" if all(checks.values()) else "fail",
                           "generated": generated, "generated_first_token_ids": generated_first_ids,
                           "maximum_batch_single_logit_drift": float((singles - logits).abs().max()),
                           "processor_class": type(processor.image_processor).__name__, "screening_config": config,
                           "model": args.model, "is_quality_or_efficacy_evidence": False}
                (out.with_suffix(".runtime.json")).write_text(json.dumps(runtime, indent=2) + "\n")
                assert all(checks.values()), runtime
            for item, values, probs, total in zip(current, logits, probabilities, mass):
                prediction = "ABCD"[int(values.argmax())]
                top = values.topk(2).values
                passed = (prediction == item["expected"] and float(probs.max()) >= config["minimum_option_probability"]
                          and float(total) >= config["minimum_total_option_mass"]
                          and float(top[0] - top[1]) >= config["minimum_logit_margin"])
                result = {**item, "prediction": prediction, "pass": passed, "option_logits": values.tolist(),
                          "option_probabilities": probs.tolist(), "total_option_probability_mass": float(total),
                          "model": args.model, "revision": cfg["revision"], "screening_config": config,
                          "dataset_sha256": config["dataset_sha256"],
                          "prompt_sha256": hashlib.sha256(PROMPT.format(role=item["role"]).encode()).hexdigest()}
                stream.write(json.dumps(result) + "\n")
                done[item["bundle_id"], item["kind"], item["mirrored"]] = result
            stream.flush()
            if start % (cfg["batch_size"] * 10) == 0:
                print(json.dumps({"model": args.model, "checks_complete": len(done),
                                  "total": len(items), "elapsed_seconds": time.monotonic() - started}), flush=True)
    grouped = defaultdict(list)
    for row in done.values():
        grouped[row["bundle_id"]].append(row)
    passed = {bid for bid, values in grouped.items() if len(values) == 4 and all(r["pass"] for r in values)}
    summary = {"checks": len(done), "groups": len(grouped), "passed_groups": len(passed),
               "passed_family_counts": dict(Counter(grouped[bid][0]["family"] for bid in passed)),
               "scope": "relative active-role pseudo labels only", "dataset_sha256": config["dataset_sha256"],
               "elapsed_seconds": time.monotonic() - started,
               "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated()}
    (out.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""重放质检首批，核对图像批次、单样本判断和真实生成答案。"""
import hashlib
import json
import os
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from audit_P051_visual import PROMPT


def main():
    config = json.loads(Path("configs/P051_tracebind.json").read_text())
    assert os.environ["CUDA_VISIBLE_DEVICES"] == config["gpu_uuid"]
    cfg = config["data"]["visual_screening"]
    run = Path("runs/P051_tracebind")
    rows = list(map(json.loads, (run / "visual_large_96_mirror.jsonl").open()))[:24]
    entries = {e["idx"]: e for e in json.loads((run / "orbits_image_manifest.json").read_text())["entries"]}
    path = cfg["models"]["large"]["path"]
    processor = AutoProcessor.from_pretrained(path, local_files_only=True, use_fast=True,
                                               min_pixels=cfg["min_pixels"], max_pixels=cfg["max_pixels"])
    processor.tokenizer.padding_side = "left"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(path, local_files_only=True, torch_dtype=torch.bfloat16,
                                                              device_map={"": 0}, attn_implementation="sdpa").eval()
    option_ids = [processor.tokenizer.encode(label, add_special_tokens=False)[0] for label in "ABCD"]
    def prepare(items):
        messages, pictures = [], []
        for item in items:
            content = []
            for label, product in zip(["Target product", "Reference A", "Reference B"], item["product_ids"]):
                content.extend([{"type": "text", "text": label + ":\n"}, {"type": "image"}])
                entry = entries[product]
                assert hashlib.sha256(Path(entry["path"]).read_bytes()).hexdigest() == entry["sha256"]
                pictures.append(Image.open(entry["path"]).convert("RGB"))
            content.append({"type": "text", "text": PROMPT.format(role=item["role"])})
            messages.append([{"role": "user", "content": content}])
        texts = [processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in messages]
        inputs = processor(text=texts, images=pictures, padding=True, return_tensors="pt").to("cuda")
        grids = inputs["image_grid_thw"].reshape(len(items), 3, 3)
        counts = grids.prod(-1).sum(-1) // model.config.vision_config.spatial_merge_size ** 2
        assert torch.equal((inputs["input_ids"] == model.config.image_token_id).sum(-1), counts)
        return inputs
    with torch.inference_mode():
        batch = model(**prepare(rows), use_cache=False, logits_to_keep=1).logits[:, -1, option_ids].float().cpu()
        previous = torch.tensor([r["option_logits"] for r in rows])
        single, generation, vocabulary_top = [], [], []
        for row in rows[:4]:
            inputs = prepare([row])
            logits = model(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1].float()
            single.append(logits[option_ids].cpu())
            top = logits.topk(5)
            vocabulary_top.append([{ "token": processor.tokenizer.decode([int(i)]), "logit": float(v)} for i, v in zip(top.indices, top.values)])
            ids = model.generate(**inputs, max_new_tokens=4, do_sample=False, repetition_penalty=1.0)
            generation.append(processor.tokenizer.decode(ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
        single = torch.stack(single)
    checks = {"all_image_token_counts_match_three_images_per_row": True,
              "batch_replay_exact": bool(torch.equal(batch, previous)),
              "batch_and_single_argmax_agree": bool(torch.equal(batch[:4].argmax(-1), single.argmax(-1))),
              "generation_is_same_option_as_single_logits": all(text.strip() == "ABCD"[int(values.argmax())] for text, values in zip(generation, single))}
    report = {"classification": "pass" if all(checks.values()) else "investigate_backend_drift", "checks": checks,
              "batch_single_max_logit_drift": float((batch[:4] - single).abs().max()), "generation": generation,
              "vocabulary_top": vocabulary_top, "processor_class": type(processor.image_processor).__name__,
              "is_data_correctness_or_efficacy_evidence": False}
    (run / "judge_runtime_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

import argparse
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from interventions import delimiter_intervention

ROOT = Path("/vepfs/vlms/1")
GPU = "GPU-7ce1c8eb-f665-99cf-2541-3c905a3f98f1"
assert os.environ.get("CUDA_VISIBLE_DEVICES") == GPU
assert Path.cwd().resolve() == ROOT
assert torch.cuda.device_count() == 1

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["baseline", "scale", "gate", "mechanism"], required=True)
parser.add_argument("--limit", type=int, default=0)
parser.add_argument("--max-pixels", type=int, default=1605632)
parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
parser.add_argument("--seed", type=int, default=17)
parser.add_argument("--output", required=True)
args = parser.parse_args()
out = Path(args.output).resolve()
assert out.is_relative_to(ROOT)
out.mkdir(parents=True, exist_ok=True)
model_path = ROOT / "models/Qwen2.5-VL-7B-Instruct"
parquets = sorted((ROOT / "data/Mantis-Eval").rglob("*.parquet"))
assert parquets
dataset = load_dataset("parquet", data_files=[str(p) for p in parquets], split="train")
if args.limit:
    dataset = dataset.select(range(min(len(dataset), args.limit)))
processor = AutoProcessor.from_pretrained(model_path, min_pixels=256*28*28,
                                          max_pixels=args.max_pixels, local_files_only=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype=getattr(torch, args.dtype), device_map={"": 0},
    attn_implementation="sdpa", local_files_only=True).eval()
metadata = vars(args) | {
    "torch": torch.__version__, "transformers": transformers.__version__,
    "gpu_uuid": GPU, "num_rows": len(dataset),
    "model_revision": json.loads((model_path / "source_manifest.json").read_text())["revision"],
    "dataset_revision": json.loads((ROOT / "data/Mantis-Eval/source_manifest.json").read_text())["revision"],
    "protocol": "DTS official prompt and parsing; SDPA; fixed scale=8 layers=0..3; sampled temperature=0.2 max_new_tokens=64",
    "source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), ROOT / "src/interventions.py"]}
}
(out / "config.json").write_text(json.dumps(metadata, indent=2))
def prompt(doc):
    question = doc["question"]
    if doc["question_type"] == "short-answer":
        text = f'Given the images, answer the following short answer vqa question:\nQ: {question}\nYou can first give your analysis, and then give your final answer as "Final Answer:"'
    else:
        text = f"{question}\nAnswer with the option's letter from the given choices directly." + str(doc["options"])
    return text.replace("<image>", "")

def inputs_for(doc):
    content = [{"type": "image", "image": img.convert("RGB"),
                "min_pixels": 256*28*28, "max_pixels": args.max_pixels} for img in doc["images"]]
    messages = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": content + [{"type": "text", "text": prompt(doc)}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    return processor(text=[text], images=images, videos=videos, padding=True, return_tensors="pt").to("cuda:0")

def parse_answer(answer):
    for marker in ["final answer:", "the answer is", "answer:"]:
        if marker in answer.lower():
            return answer[answer.lower().index(marker)+len(marker):].strip()
    return answer

def score(doc, raw):
    answer = parse_answer(raw)
    ref = doc["answer"].strip("()\n ")
    if doc["question_type"] == "short-answer":
        return answer, answer.lower() == ref.lower()
    if len(ref) != 1:
        ref = next(c for c in ref if c.isalpha())
    selected = answer[8] if re.match(r"Answer: [A-Z]", answer) else next((c.upper() for c in answer if c.isalpha()), None)
    if selected and ord(selected)-ord("A") < len(doc["options"]):
        return selected, selected == ref.upper()
    idx = ord(ref.upper())-ord("A")
    if idx >= len(doc["options"]):
        return raw, False
    option = doc["options"][idx]
    for prefix in [ref+".", ref+":", "("+ref+")"]:
        if option.startswith(prefix):
            option = option[len(prefix):]
            break
    return raw, raw.strip() == option.strip()

results_path = out / "predictions.jsonl"
done = {}
if results_path.exists():
    for line in results_path.read_text().splitlines():
        r = json.loads(line)
        done[r["id"]] = r
torch.backends.cuda.matmul.allow_tf32 = False
with torch.inference_mode(), results_path.open("a", buffering=1) as handle:
    for index, doc in enumerate(dataset):
        if doc["id"] in done:
            continue
        started = time.monotonic()
        inputs = inputs_for(doc)
        torch.cuda.reset_peak_memory_stats()
        if args.mode == "mechanism":
            logits = {}
            for mode in ["baseline", "scale", "gate"]:
                with delimiter_intervention(model, inputs["input_ids"], mode):
                    logits[mode] = model(**inputs, use_cache=False).logits[:, -1, :].float().cpu()
            a, b = logits["scale"], logits["gate"]
            rec = {"id":doc["id"],"max_abs_error":(a-b).abs().max().item(),
                   "relative_l2_error":((a-b).norm()/a.norm()).item(),
                   "same_top1":bool(a.argmax() == b.argmax()),
                   "baseline_top1":processor.tokenizer.decode(logits["baseline"].argmax(-1)),
                   "scale_top1":processor.tokenizer.decode(a.argmax(-1)),
                   "gate_top1":processor.tokenizer.decode(b.argmax(-1))}
        else:
            torch.manual_seed(args.seed+index)
            random.seed(args.seed+index)
            np.random.seed(args.seed+index)
            with delimiter_intervention(model, inputs["input_ids"], args.mode):
                generated = model.generate(**inputs, do_sample=True, temperature=0.2, top_p=None,
                                           max_new_tokens=64, num_beams=1, use_cache=True,
                                           eos_token_id=processor.tokenizer.eos_token_id,
                                           pad_token_id=processor.tokenizer.pad_token_id)
            suffix = generated[:, inputs["input_ids"].shape[1]:]
            raw = processor.batch_decode(suffix, skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)[0]
            answer, correct = score(doc, raw)
            rec = {"id":doc["id"],"category":doc["category"],"question_type":doc["question_type"],
                   "answer":doc["answer"],"raw_prediction":raw,"parsed_prediction":answer,
                   "correct":bool(correct),"output_tokens":suffix.shape[1],
                   "hit_token_limit":suffix.shape[1] == 64}
        torch.cuda.synchronize()
        rec.update({"input_tokens":inputs["input_ids"].shape[1],"images":len(doc["images"]),
                    "seconds":time.monotonic()-started,"peak_memory_gb":torch.cuda.max_memory_allocated()/1e9})
        handle.write(json.dumps(rec, ensure_ascii=False)+"\n")
        done[doc["id"]] = rec
        print(json.dumps({"progress":f"{index+1}/{len(dataset)}","mode":args.mode}|rec,ensure_ascii=False),flush=True)
        del inputs
rows = list(done.values())
summary = {"n":len(rows),"mode":args.mode,"complete":len(rows)==len(dataset)}
if args.mode == "mechanism":
    summary.update({"same_top1_rate":sum(r["same_top1"] for r in rows)/len(rows),
                    "max_abs_error":max(r["max_abs_error"] for r in rows),
                    "max_relative_l2_error":max(r["relative_l2_error"] for r in rows)})
else:
    summary["accuracy"] = sum(r["correct"] for r in rows)/len(rows)
    summary["token_limit_count"] = sum(r["hit_token_limit"] for r in rows)
(out / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary), flush=True)

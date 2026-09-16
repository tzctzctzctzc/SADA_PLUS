import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from boundary_interventions import boundary_intervention

ROOT = Path("/vepfs/vlms/1")
GPU = "GPU-7ce1c8eb-f665-99cf-2541-3c905a3f98f1"
assert Path.cwd().resolve() == ROOT
assert os.environ.get("CUDA_VISIBLE_DEVICES") == GPU and torch.cuda.device_count() == 1
config = json.loads((ROOT / "configs/P003_boundary_counterfactual.json").read_text())
out = ROOT / "runs/P003_boundary_probe"
out.mkdir(exist_ok=True)
files = list((ROOT / "data/Mantis-Eval").rglob("*.parquet"))
ds = load_dataset("parquet", data_files={"test":list(map(str,files))}, split="test")
path = ROOT / "models/Qwen2.5-VL-7B-Instruct"
processor = AutoProcessor.from_pretrained(path, min_pixels=256*28*28, max_pixels=config["max_pixels"], local_files_only=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(path, torch_dtype=torch.float32,
    device_map={"":0},attn_implementation="sdpa",local_files_only=True).eval()
torch.backends.cuda.matmul.allow_tf32 = False
config["source_hashes"] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), ROOT/"src/boundary_interventions.py",ROOT/"src/interventions.py"]}
config["torch"] = torch.__version__
config["gpu_uuid"] = GPU
(out/"config.json").write_text(json.dumps(config,indent=2))

def make_inputs(images):
    content = [{"type":"image","image":im.convert("RGB"),"min_pixels":256*28*28,"max_pixels":config["max_pixels"]} for im in images]
    messages = [{"role":"system","content":"You are a helpful assistant."},
        {"role":"user","content":content+[{"type":"text","text":config["prompt"]}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ims, vids = process_vision_info(messages)
    return processor(text=[text],images=ims,videos=vids,return_tensors="pt").to("cuda:0")

def run(inputs, mode):
    ids = inputs["input_ids"][0]
    starts = (ids == model.config.vision_start_token_id).nonzero(as_tuple=True)[0]
    ends = (ids == model.config.vision_end_token_id).nonzero(as_tuple=True)[0]
    assert len(starts) == len(ends) == 2
    captures = []
    def capture(module, args, output):
        values = torch.stack([output[0,starts[0]], output[0,starts[1]],
            output[0,ends[1]], output[0,starts[1]+1:ends[1]].mean(0), output[0,-1]])
        captures.append(values.float().cpu())
    with boundary_intervention(model, inputs["input_ids"], mode):
        handles = [b.input_layernorm.register_forward_hook(capture) for b in model.model.layers]
        try:
            logits = model(**inputs, use_cache=False).logits[0,-1].float().cpu()
        finally:
            for h in handles:
                h.remove()
    return logits, torch.stack(captures)

def js(a,b):
    la,lb = a.log_softmax(-1),b.log_softmax(-1)
    lm = torch.logaddexp(la,lb)-np.log(2)
    return float(0.5*((la.exp()*(la-lm)).sum()+(lb.exp()*(lb-lm)).sum()))

def cosine_drift(a,b):
    return (1-torch.nn.functional.cosine_similarity(a,b,dim=-1)).clamp_min(0).tolist()

result_path = out/"results.jsonl"
rows = [json.loads(x) for x in result_path.read_text().splitlines()] if result_path.exists() else []
done = {(r["id"],r["mode"]) for r in rows}
with torch.inference_mode(),result_path.open("a",buffering=1) as f:
    for i in range(8):
        doc,donor = ds[i],ds[i+1]
        original = [im.convert("RGB") for im in doc["images"][:2]]
        replacements = [donor["images"][j].convert("RGB").resize(original[j].size) for j in range(2)]
        variants = {"original":original,"prefix_replaced":[replacements[0],original[1]],
            "target_replaced":[original[0],replacements[1]]}
        inputs = {k:make_inputs(v) for k,v in variants.items()}
        assert all(torch.equal(inputs["original"]["input_ids"],v["input_ids"]) for v in inputs.values())
        assert all(torch.equal(inputs["original"]["image_grid_thw"],v["image_grid_thw"]) for v in inputs.values())
        for mode in config["conditions"]:
            if (doc["id"],mode) in done:
                continue
            started = time.monotonic()
            measurements = {k:run(v,mode) for k,v in inputs.items()}
            base_logits,base_states = measurements["original"]
            rec = {"id":doc["id"],"donor_id":donor["id"],"mode":mode,"input_tokens":inputs["original"]["input_ids"].shape[1]}
            for variant in ("prefix_replaced","target_replaced"):
                logits,states = measurements[variant]
                rec[variant] = {"js":js(base_logits,logits),"same_top1":bool(base_logits.argmax()==logits.argmax()),
                    "cosine_drift_by_layer":cosine_drift(base_states,states)}
            rec["top1"] = {k:processor.tokenizer.decode(v[0].argmax().reshape(1)) for k,v in measurements.items()}
            rec["seconds"] = time.monotonic()-started
            rec["peak_memory_gb"] = torch.cuda.max_memory_allocated()/1e9
            f.write(json.dumps(rec)+"\n")
            rows.append(rec)
            print(json.dumps({k:v for k,v in rec.items() if k not in ("prefix_replaced","target_replaced")}),flush=True)
        del inputs
summary={"n":len(rows),"complete":len(rows)==8*len(config["conditions"]),"conditions":{}}
for mode in config["conditions"]:
    subset = [r for r in rows if r["mode"]==mode]
    summary["conditions"][mode]={v:{"mean_js":float(np.mean([r[v]["js"] for r in subset])),
        "median_js":float(np.median([r[v]["js"] for r in subset]))} for v in ("prefix_replaced","target_replaced")}
(out/"summary.json").write_text(json.dumps(summary,indent=2))
print(json.dumps(summary),flush=True)

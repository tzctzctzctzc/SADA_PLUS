"""Four-case, post-hoc diagnostic; clause scores are not an evaluation method."""
import hashlib, json, os, time
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

ROOT=Path("/vepfs/vlms/1")
assert Path.cwd().resolve()==ROOT
cfg_path=Path("configs/P029_clause_diagnostic.json")
cfg=json.loads(cfg_path.read_text())
assert os.environ["CUDA_VISIBLE_DEVICES"]==cfg["gpu_uuid"]
assert torch.cuda.device_count()==1
torch.backends.cuda.matmul.allow_tf32=False
data={r["id"]:r for r in json.loads(Path(cfg["data_records"]).read_text())}
model_path=ROOT/"models"/cfg["model"]
proc=AutoProcessor.from_pretrained(model_path,local_files_only=True,
    min_pixels=cfg["min_pixels"],max_pixels=cfg["max_pixels"])
model=Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path,
    local_files_only=True,torch_dtype="auto",device_map={"":0},
    attn_implementation="sdpa").eval()
ids=[proc.tokenizer.encode(t,add_special_tokens=False) for t in ["True","False"]]
assert all(len(x)==1 for x in ids)
ids=[x[0] for x in ids]
out=ROOT/"runs/P029_clause_diagnostic";out.mkdir(exist_ok=True)
prior={(r["base_id"],r["image_index"]):r for r in map(json.loads,Path("runs/P025_pointwise_imagecode/predictions.jsonl").read_text().splitlines())}
results=[];replay_errors=[]
with torch.inference_mode(),(out/"predictions.jsonl").open("w") as f:
    for case in cfg["cases"]:
        row=data[case["id"]]
        assert row["caption"]==case["caption"]
        for ci,claim in enumerate([case["caption"]]+case["clauses"]):
            for ix in case["image_indices"]:
                image=Image.open(ROOT/row["images"][ix]["path"]).convert("RGB")
                prompt=cfg["matching_prompt"].replace("{caption}",claim)
                text=proc.apply_chat_template([{"role":"user","content":[
                    {"type":"image"},{"type":"text","text":prompt}]}],
                    tokenize=False,add_generation_prompt=True)
                inputs=proc(text=[text],images=[image],return_tensors="pt")
                inputs={k:v.to("cuda:0",dtype=model.dtype if v.is_floating_point() else None)
                        for k,v in inputs.items()}
                start=time.monotonic()
                logits=model(**inputs,use_cache=False).logits[0,-1,ids].float().cpu().tolist()
                result={"id":case["id"],"claim_index":ci,"claim":claim,
                        "image_index":ix,"gold":case["gold"],
                        "logits":logits,"margin":logits[0]-logits[1],
                        "image_grid":inputs["image_grid_thw"].cpu().tolist(),
                        "input_tokens":inputs["input_ids"].shape[1],
                        "seconds":time.monotonic()-start}
                if ci==0:
                    ref=prior[case["id"],ix]
                    assert result["image_grid"]==ref["image_grid_thw"]
                    assert result["input_tokens"]==ref["matching_input_tokens"]
                    replay_errors.append(max(abs(a-b) for a,b in zip(logits,ref["true_false_logits"])))
                f.write(json.dumps(result)+"\n");f.flush();results.append(result)
expected=sum(2*(1+len(c["clauses"])) for c in cfg["cases"])
assert len(results)==expected
report={"complete":True,"n_records":len(results),"n_cases":len(cfg["cases"]),
        "config_sha256":hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
        "runner_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "full_caption_replay_max_logit_error":max(replay_errors),
        "interpretation":cfg["interpretation"],"limits":cfg["limits"],"cases":[]}
for c in cfg["cases"]:
    rr=[r for r in results if r["id"]==c["id"]]
    detail={"id":c["id"],"gold":c["gold"],"error_image":c["image_indices"][1],"claims":[]}
    for ci in range(1+len(c["clauses"])):
        pair={r["image_index"]:r for r in rr if r["claim_index"]==ci}
        a,b=[pair[ix] for ix in c["image_indices"]]
        detail["claims"].append({"text":a["claim"],"claim_index":ci,
            "gold_margin":a["margin"],"error_margin":b["margin"],
            "gold_minus_error":a["margin"]-b["margin"]})
    report["cases"].append(detail)
Path("results/P029_clause_diagnostic.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))

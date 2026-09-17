"""固定 PinPoint 候选上的 Coral / 同底座迁移基线；只取官方末 token 表征。"""
import argparse, hashlib, json, os, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, GenerationConfig, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

ROOT=Path("/vepfs/vlms/1")
assert Path.cwd().resolve()==ROOT
CFG_PATH=ROOT/"configs/P031_coral_transfer.json"
cfg=json.loads(CFG_PATH.read_text())
parser=argparse.ArgumentParser()
parser.add_argument("--preflight-only",action="store_true")
args=parser.parse_args()
source=ROOT/cfg["cohort_config"]
assert hashlib.sha256(source.read_bytes()).hexdigest()==cfg["cohort_sha256"]
cohort=json.loads(source.read_text())
rows=cohort["records"]
images={x["signature"]:x for x in json.loads((ROOT/"data/PinPoint/P030_image_manifest.json").read_text())["entries"]}
out=ROOT/"runs/P031_coral_transfer"
out.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(8)
candidate_ids=sorted({s for row in rows for s in row["candidates_to_score"]})
tasks=[{"key":"candidate_"+s,"signatures":[s],"kind":"candidate"} for s in candidate_ids]
for row in rows:
    for condition in cfg["conditions"]:
        sigs={"single":row["references"][:1],"dual":row["references"],
              "sham":[row["references"][0],row["sham_reference2"]],"text_only":[]}[condition]
        tasks.append({"key":"query_"+row["id"]+"_"+condition,"signatures":sigs,
                      "kind":"query","condition":condition,"instruction":row["instruction"]})
assert len(tasks)==cfg["expected_embeddings_per_model"]
def message(task):
    if task["kind"]=="candidate":
        return [{"role":"user","content":[{"type":"text","text":"Represent the given product: "},{"type":"image"}]}]
    if task["condition"]=="text_only":
        return [{"role":"user","content":[{"type":"text","text":task["instruction"]}]}]
    # 参考图在指令中首次出现的位置插入，保持原有词语与来源编号。
    import re
    content=[];used=set();pos=0
    text=task["instruction"]
    for match in re.finditer(r"<img([12])>",text):
        if match.start()>pos:content.append({"type":"text","text":text[pos:match.start()]})
        index=int(match.group(1))
        present=index<=len(task["signatures"])
        content.append({"type":"text","text":f"<Product {index}> "})
        if present and index not in used:
            content.extend([{"type":"image"},{"type":"text","text":f" </Product {index}>"}])
            used.add(index)
        pos=match.end()
    if pos<len(text):content.append({"type":"text","text":text[pos:]})
    assert len(used)==len(task["signatures"])
    return [{"role":"user","content":content}]
def ordered_sigs(task):
    if task["kind"]=="candidate":return task["signatures"]
    import re
    used=[]
    for k in re.findall(r"<img([12])>",task["instruction"]):
        i=int(k)-1
        if i<len(task["signatures"]) and i not in used:used.append(i)
    return [task["signatures"][i] for i in used]
verified_images=set()
def load_image(sig):
    p=ROOT/"data/PinPoint/images"/(sig+".jpg")
    if sig not in verified_images:
        assert hashlib.sha256(p.read_bytes()).hexdigest()==images[sig]["sha256"],sig
        verified_images.add(sig)
    with Image.open(p) as im:return im.convert("RGB")
def prepare(processor,batch):
    texts=[processor.apply_chat_template(message(t),tokenize=False,add_generation_prompt=True) for t in batch]
    pics=[load_image(s) for t in batch for s in ordered_sigs(t)]
    inputs=processor(text=texts,images=pics or None,padding=True,return_tensors="pt")
    grids=inputs.get("image_grid_thw")
    if pics:
        assert len(grids)==len(pics)
        expected=sum(int(torch.prod(g))//4 for g in grids)
        assert int((inputs["input_ids"]==151655).sum())==expected
    assert bool(inputs["attention_mask"][:,-1].all())
    return inputs
def verify_weights(model_cfg):
    directory=ROOT/model_cfg["path"]
    manifest=json.loads((directory/"source_manifest.json").read_text())
    assert manifest["revision"]==model_cfg["revision"]
    checked=[]
    for entry in manifest["files"]:
        if not entry["path"].endswith(".safetensors"):continue
        p=directory/entry["path"];assert p.stat().st_size==entry["size"]
        h=hashlib.sha256()
        with p.open("rb") as f:
            while chunk:=f.read(8*1024*1024):h.update(chunk)
        assert h.hexdigest()==entry["lfs"]["sha256"]
        checked.append({"path":entry["path"],"sha256":h.hexdigest()})
    assert checked
    return checked
def compatible_model_config(model_cfg):
    raw=json.loads((ROOT/model_cfg["path"]/"config.json").read_text())
    if isinstance(raw.get("text_config"),dict):
        nested=raw.pop("text_config")
        if "tie_word_embeddings" not in raw:
            raw["tie_word_embeddings"]=nested["tie_word_embeddings"]
        compat=Qwen2_5_VLConfig.from_dict(raw)
        GenerationConfig.from_model_config(compat)
        return compat,"Flattened the redundant nested text_config emitted by a newer transformers version; copied its tie_word_embeddings flag and kept the same Qwen2.5-VL fields and weights."
    return None,None
metadata={"config_sha256":hashlib.sha256(CFG_PATH.read_bytes()).hexdigest(),
          "runner_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "n_tasks":len(tasks)}
meta_path=out/"metadata.json"
if meta_path.exists():
    prior=json.loads(meta_path.read_text())
    assert prior["config_sha256"]==metadata["config_sha256"] and prior["n_tasks"]==metadata["n_tasks"]
    if prior["runner_sha256"]!=metadata["runner_sha256"]:
        retry={**metadata,"prior_runner_sha256":prior["runner_sha256"],
               "reason":"Resume completed base3b embeddings after Coral config compatibility failure; no Coral embeddings or final analysis existed."}
        (out/"retry_metadata.json").write_text(json.dumps(retry,indent=2)+"\n")
else:meta_path.write_text(json.dumps(metadata,indent=2)+"\n")
if args.preflight_only:
    checks={}
    for name,model_cfg in cfg["models"].items():
        compatible_model_config(model_cfg)
        processor=AutoProcessor.from_pretrained(ROOT/model_cfg["path"],local_files_only=True,
                    min_pixels=cfg["min_pixels"],max_pixels=cfg["max_pixels"])
        processor.tokenizer.padding_side="left"
        # 包含第二图先出现的例子，检验图片列表按提示中的出现顺序排列。
        samples=[tasks[0]]+[t for t in tasks if t["kind"]=="query" and t["key"].startswith("query_"+rows[0]["id"]+"_")]
        reverse=next(t for t in tasks if t["kind"]=="query" and t["condition"]=="dual" and t["instruction"].find("<img2>")<t["instruction"].find("<img1>"))
        samples.append(reverse)
        checks[name]=[{"key":t["key"],"tokens":int(prepare(processor,[t])["attention_mask"].sum()),
                      "image_order":ordered_sigs(t)} for t in samples]
    (out/"cpu_preflight.json").write_text(json.dumps(checks,indent=2)+"\n")
    print("CPU_PREFLIGHT_OK",len(checks),flush=True)
    raise SystemExit(0)
assert os.environ["CUDA_VISIBLE_DEVICES"]==cfg["gpu_uuid"]
assert torch.cuda.device_count()==1
assert "A100" in torch.cuda.get_device_name(0)
for name,model_cfg in cfg["models"].items():
    print("MODEL_START",name,flush=True)
    checked=verify_weights(model_cfg)
    folder=out/name;folder.mkdir(exist_ok=True)
    processor=AutoProcessor.from_pretrained(ROOT/model_cfg["path"],local_files_only=True,
                min_pixels=cfg["min_pixels"],max_pixels=cfg["max_pixels"])
    processor.tokenizer.padding_side="left"
    compat_config,config_compatibility=compatible_model_config(model_cfg)
    load_kwargs={"local_files_only":True,"torch_dtype":torch.bfloat16,
                 "attn_implementation":"sdpa","device_map":{"":"cuda:0"},
                 "output_loading_info":True}
    if compat_config is not None:load_kwargs["config"]=compat_config
    model,loading=Qwen2_5_VLForConditionalGeneration.from_pretrained(
        ROOT/model_cfg["path"],**load_kwargs)
    assert not loading["missing_keys"] and not loading.get("mismatched_keys"),loading
    assert not loading["unexpected_keys"],loading
    model.eval()
    capture={}
    def last_hidden_hook(module,inputs):
        capture["last"]=inputs[0][:,-1,:].detach()
        return (inputs[0][:,-1:,:],)
    handle=model.lm_head.register_forward_pre_hook(last_hidden_hook)
    sample=prepare(processor,[tasks[0]]).to("cuda")
    with torch.inference_mode():
        reference=model(**sample,return_dict=True,output_hidden_states=True,use_cache=False)
    error=float((capture["last"]-reference.hidden_states[-1][:,-1,:]).abs().max())
    assert error==0
    solo=capture["last"].clone()
    del reference,sample
    with torch.inference_mode():
        model(**prepare(processor,tasks[:cfg["batch_size"]]).to("cuda"),use_cache=False)
    batch_cos=float(F.cosine_similarity(solo.float(),capture["last"][:1].float()).item())
    assert batch_cos>0.995,batch_cos
    (folder/"protocol_smoke.json").write_text(json.dumps({
        "hook_vs_official_last_hidden_max_error":error,"single_vs_batch_cosine":batch_cos,
        "loading_info":loading,"weights":checked,
        "config_compatibility":config_compatibility},indent=2)+"\n")
    pending=[t for t in tasks if not (folder/(t["key"]+".npy")).exists()]
    for start in range(0,len(pending),cfg["batch_size"]):
        batch=pending[start:start+cfg["batch_size"]];begin=time.perf_counter()
        inputs=prepare(processor,batch).to("cuda")
        with torch.inference_mode():model(**inputs,use_cache=False)
        embeddings=capture["last"].float().cpu().numpy()
        assert embeddings.shape==(len(batch),2048) and np.isfinite(embeddings).all()
        elapsed=time.perf_counter()-begin
        with (folder/"trace.jsonl").open("a") as log:
            for t,emb in zip(batch,embeddings):
                path=folder/(t["key"]+".npy")
                with path.with_suffix(".tmp").open("wb") as f:np.save(f,emb)
                path.with_suffix(".tmp").replace(path)
                log.write(json.dumps({"key":t["key"],"image_order":ordered_sigs(t),
                    "batch_seconds_per_item":elapsed/len(batch)})+"\n")
        if start%80==0:print(name,"done",len(tasks)-len(pending)+start+len(batch),"/",len(tasks),flush=True)
        del inputs
    assert all((folder/(t["key"]+".npy")).exists() for t in tasks)
    handle.remove();del model,capture,solo
    torch.cuda.empty_cache()
    print("MODEL_DONE",name,flush=True)
(out/"complete.json").write_text(json.dumps({"complete":True,"n_models":len(cfg["models"]),"n_embeddings_per_model":len(tasks)})+"\n")

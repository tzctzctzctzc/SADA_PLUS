"""Frozen PinPoint input-completeness controls; all inference is on physical GPU 3."""
import ast,copy,functools,hashlib,io,json,os,time
from pathlib import Path
from typing import Dict,List
import torch,transformers
from PIL import Image
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration

ROOT=Path("/vepfs/vlms/1");assert Path.cwd().resolve()==ROOT
cp=Path("configs/P030_pinpoint_references.json");cfg=json.loads(cp.read_text())
assert os.environ["CUDA_VISIBLE_DEVICES"]==cfg["gpu_uuid"]
assert torch.cuda.device_count()==1
torch.set_num_threads(8);torch.manual_seed(17030)
torch.backends.cuda.matmul.allow_tf32=False
image_manifest_path=Path("data/PinPoint/P030_image_manifest.json")
imeta=json.loads(image_manifest_path.read_text())
assert imeta["complete"]
assert imeta["config_sha256"]==hashlib.sha256(cp.read_bytes()).hexdigest()
images={r["signature"]:r for r in imeta["entries"]}
assert set(images)==set(cfg["image_signatures"])
sm=json.loads(Path("literature/PinPoint_source_manifest.json").read_text())
source=Path("third_party/pinpoint-dataset/rerank.py")
assert hashlib.sha256(source.read_bytes()).hexdigest()==next(r["sha256"] for r in sm["files"] if r["path"]=="rerank.py")
tree=ast.parse(source.read_text());ns={"Dict":Dict,"List":List}
nodes=[n for n in tree.body if (isinstance(n,ast.FunctionDef) and n.name=="build_messages") or
       (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {"SYSTEM_PROMPT","BASE_INSTRUCTIONS"} for t in n.targets))]
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),"exec"),ns)
out=Path("runs/P030_pinpoint_references");out.mkdir(exist_ok=True)
model_path=ROOT/"models"/cfg["model"]
assert json.loads((model_path/"source_manifest.json").read_text())["revision"]==cfg["model_revision"]
proc=AutoProcessor.from_pretrained(model_path,local_files_only=True,
    min_pixels=cfg["min_pixels"],max_pixels=cfg["max_pixels"])
proc.tokenizer.padding_side="left"
model=Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path,local_files_only=True,
    torch_dtype=torch.bfloat16,device_map={"":0},attn_implementation=cfg["backend"]).eval()
token_ids=[proc.tokenizer.encode(t,add_special_tokens=False) for t in ["True","False"]]
assert all(len(x)==1 for x in token_ids);token_ids=[x[0] for x in token_ids]
metadata={"config_sha256":hashlib.sha256(cp.read_bytes()).hexdigest(),
          "runner_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "image_manifest_sha256":hashlib.sha256(image_manifest_path.read_bytes()).hexdigest(),
          "torch":torch.__version__,"transformers":transformers.__version__,
          "model_revision":cfg["model_revision"],"token_ids":token_ids,
          "last_token_optimization":"Installed transformers lacks logits_to_keep; a verified lm_head pre-hook slices only the final hidden state. No labels/loss are used.",
          "batch_size":cfg["batch_size"]}
mp=out/"metadata.json"
if mp.exists():assert json.loads(mp.read_text())==metadata
else:mp.write_text(json.dumps(metadata,indent=2)+"\n")
@functools.lru_cache(maxsize=64)
def image_for(sig):
    e=images[sig];b=(ROOT/e["path"]).read_bytes()
    assert hashlib.sha256(b).hexdigest()==e["sha256"]
    return Image.open(io.BytesIO(b)).convert("RGB")
def payload(row,condition,candidate):
    messages=copy.deepcopy(ns["build_messages"](row["instruction"],row["references"][0],candidate))
    if condition=="released_single":return messages
    content=[];ordinal=0
    for item in messages[1]["content"]:
        if item["type"]=="text":
            item["text"]=item["text"].replace("**Reference Image:**","**Reference Image <img1>:**")
            content.append(item)
        elif item["type"]=="image_url":
            if ordinal==0:
                content.append(item)
                if condition in ["labelled_dual","labelled_sham"]:
                    sig=row["references"][1] if condition=="labelled_dual" else row["sham_reference2"]
                    content.extend([{"type":"text","text":"\n**Reference Image <img2>:**\n"},
                                    {"type":"image_url","image_url":{"url":sig}}])
            else:
                content.extend([{"type":"text","text":"\n**Candidate Image:**\n"},item])
            ordinal+=1
    assert ordinal==2
    messages[1]["content"]=content
    return messages
def prepare(items):
    texts=[];flat_images=[];counts=[];details=[]
    for item in items:
        messages=payload(item["row"],item["condition"],item["candidate"])
        sigs=[];converted=[]
        for message in messages:
            if isinstance(message["content"],str):converted.append(message);continue
            content=[]
            for part in message["content"]:
                if part["type"]=="image_url":
                    sig=part["image_url"]["url"];sigs.append(sig)
                    flat_images.append(image_for(sig));content.append({"type":"image"})
                else:content.append(part)
            converted.append({"role":message["role"],"content":content})
        expected=3 if item["condition"] in ["labelled_dual","labelled_sham"] else 2
        assert len(sigs)==expected
        texts.append(proc.apply_chat_template(converted,tokenize=False,add_generation_prompt=True))
        counts.append(len(sigs))
        details.append({"image_signatures":sigs,"payload_sha256":hashlib.sha256(json.dumps(messages,sort_keys=True).encode()).hexdigest()})
    batch=proc(text=texts,images=flat_images,padding=True,return_tensors="pt")
    grid=batch["image_grid_thw"];assert len(grid)==sum(counts)
    offset=0
    for i,n in enumerate(counts):
        segment=grid[offset:offset+n];offset+=n
        nt=int((segment.prod(dim=1)//4).sum())
        assert int((batch["input_ids"][i]==model.config.image_token_id).sum())==nt
        assert int(batch["attention_mask"][i,-1])==1
        details[i].update(image_grids=segment.tolist(),image_tokens=nt,
                          input_tokens=int(batch["attention_mask"][i].sum()))
    return {k:v.to("cuda:0",dtype=model.dtype if v.is_floating_point() else None) for k,v in batch.items()},details
def logits_for(batch):
    return model(**batch,use_cache=False).logits[:,-1,token_ids].float().cpu()
instances=[{"row":r,"condition":c,"candidate":sig,"key":r["id"]+"|"+c+"|"+sig}
           for r in cfg["records"] for c in cfg["conditions"] for sig in r["candidates_to_score"]]
assert len(instances)==cfg["expected_records"]
rp=out/"predictions.jsonl"
done={r["key"] for r in map(json.loads,rp.read_text().splitlines())} if rp.exists() else set()
assert done<=set(i["key"] for i in instances)
with torch.inference_mode():
    single,single_details=prepare(instances[:1])
    full_logits=logits_for(single)
    hook=model.lm_head.register_forward_pre_hook(lambda module,args:(args[0][:,-1:,:],))
    short_logits=logits_for(single)
    batch,batch_details=prepare(instances[:cfg["batch_size"]])
    batched_logits=logits_for(batch)
    assert single_details[0]==batch_details[0]
    smoke={"full_vs_last_head_max_error":float((full_logits-short_logits).abs().max()),
           "single_vs_batch_max_error_first_example":float((short_logits-batched_logits[:1]).abs().max()),
           "first_example_single_logits":short_logits[0].tolist(),
           "first_example_batch_logits":batched_logits[0].tolist(),
           "first_batch_image_counts":[len(d["image_grids"]) for d in batch_details]}
    # This is a numerical check, not a claim of bitwise reproducibility across GEMM shapes.
    assert max(smoke["full_vs_last_head_max_error"],smoke["single_vs_batch_max_error_first_example"])<=0.5
    (out/"protocol_smoke.json").write_text(json.dumps(smoke,indent=2)+"\n")
    print("protocol smoke",smoke,flush=True)
    del single,batch,full_logits,short_logits,batched_logits
    with rp.open("a",buffering=1) as f:
        for start in range(0,len(instances),cfg["batch_size"]):
            items=[i for i in instances[start:start+cfg["batch_size"]] if i["key"] not in done]
            if not items:continue
            tick=time.monotonic();batch,details=prepare(items);values=logits_for(batch)
            elapsed=time.monotonic()-tick
            for item,detail,value in zip(items,details,values.tolist()):
                r={"key":item["key"],"id":item["row"]["id"],"condition":item["condition"],
                   "candidate":item["candidate"],"true_false_logits":value,
                   "margin":value[0]-value[1],"pipeline_seconds_per_item":elapsed/len(items),**detail}
                f.write(json.dumps(r)+"\n");done.add(item["key"])
            del batch,values
            if len(done)%80==0:print("completed",len(done),"/",len(instances),flush=True)
    hook.remove()
assert len(done)==len(instances)
(out/"summary.json").write_text(json.dumps({"complete":True,"n_records":len(done),
    "max_gpu_allocated_bytes":torch.cuda.max_memory_allocated()},indent=2)+"\n")
print("P030 scoring complete",len(done),flush=True)

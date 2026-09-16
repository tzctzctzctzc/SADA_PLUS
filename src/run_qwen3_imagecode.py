"""Frozen official Qwen3-VL-Reranker baseline at default and bounded image budgets."""
import os
os.environ["HF_HUB_OFFLINE"]="1"
os.environ["TRANSFORMERS_OFFLINE"]="1"
import hashlib, importlib.util, json, time
from pathlib import Path
import torch

ROOT=Path("/vepfs/vlms/1")
assert Path.cwd().resolve()==ROOT
cfg=json.loads(Path("configs/P027_qwen3_imagecode.json").read_text())
assert os.environ["CUDA_VISIBLE_DEVICES"]==cfg["gpu_uuid"]
assert torch.cuda.device_count()==1
torch.set_num_threads(8)
torch.manual_seed(17027)
source_meta=json.loads(Path("literature/Qwen3-VL-Reranker_source.json").read_text())
source=ROOT/"third_party/Qwen3-VL-Embedding"/source_meta["file"]
assert hashlib.sha256(source.read_bytes()).hexdigest()==source_meta["sha256"]
verification=json.loads(Path("models/Qwen3-VL-Reranker-2B/verification.json").read_text())
assert verification["complete"] and verification["revision"]==cfg["model_revision"]
spec=importlib.util.spec_from_file_location("official_qwen3_reranker",source)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
data_path=Path("runs/P023_data/records.json")
data=[r for r in json.loads(data_path.read_text()) if r["split"]=="evaluation"]
assert len(data)==256
out=Path("runs/P027_qwen3_imagecode");out.mkdir(exist_ok=True)
metadata={"config":cfg,"data_sha256":hashlib.sha256(data_path.read_bytes()).hexdigest(),
          "runner_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "source":source_meta,"torch":torch.__version__}
(out/"metadata.json").write_text(json.dumps(metadata,indent=2))
model=module.Qwen3VLReranker(str(ROOT/"models/Qwen3-VL-Reranker-2B"),
                           torch_dtype=torch.bfloat16,attn_implementation="sdpa",local_files_only=True)
margins=[];grids=[]
def capture_margin(mod,args,result):
    margins.extend(result.detach().float().flatten().cpu().tolist())
hook=model.score_linear.register_forward_hook(capture_margin)
original_tokenize=model.tokenize
def checked_tokenize(*args,**kwargs):
    inputs=original_tokenize(*args,**kwargs)
    # The official helper falls back to NULL on image-loading errors; reject that path.
    assert "image_grid_thw" in inputs and tuple(inputs["image_grid_thw"].shape)==(1,3)
    assert inputs["input_ids"].shape[1]<model.max_length
    grid=inputs["image_grid_thw"][0].tolist()
    expected=int(inputs["image_grid_thw"].prod().item())//4
    actual=int((inputs["input_ids"]==model.model.config.image_token_id).sum().item())
    assert actual==expected and expected>0
    grids.append({"grid":grid,"image_tokens":expected,"total_tokens":inputs["input_ids"].shape[1]})
    return inputs
model.tokenize=checked_tokenize
with (out/"predictions.jsonl").open("w") as f:
    for setting in cfg["settings"]:
        model.min_pixels=setting["min_pixels"];model.max_pixels=setting["max_pixels"]
        for i,row in enumerate(data):
            margins.clear();grids.clear()
            start=time.monotonic()
            scores=model.process({"instruction":cfg["instruction"],"query":{"text":row["caption"]},
                                  "documents":[{"image":str(ROOT/im["path"])} for im in row["images"]]})
            assert len(scores)==len(margins)==len(grids)==10
            assert all(torch.isfinite(torch.tensor(scores))) and all(torch.isfinite(torch.tensor(margins)))
            result={"id":row["id"],"setting":setting["name"],"gold":row["gold"],
                    "scores":scores,"pre_sigmoid_margins":list(margins),"grids":list(grids),
                    "seconds":time.monotonic()-start}
            f.write(json.dumps(result)+"\n");f.flush()
            if (i+1)%32==0:print(setting["name"],i+1,len(data),flush=True)
hook.remove()
(out/"summary.json").write_text(json.dumps({"complete":True,"n_records":len(data)*len(cfg["settings"]),
                                            "max_gpu_allocated_bytes":torch.cuda.max_memory_allocated()},indent=2))

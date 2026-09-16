"""同候选比较；float32 余弦为主，同时保留模型卡的 BF16 打分口径。"""
import ast,hashlib,json
from pathlib import Path
from typing import Dict,List
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path("/vepfs/vlms/1");assert Path.cwd().resolve()==ROOT
torch.set_num_threads(8)
cfg=json.loads(Path("configs/P031_coral_transfer.json").read_text())
cohort=json.loads(Path(cfg["cohort_config"]).read_text());rows=cohort["records"]
source=Path("third_party/pinpoint-dataset/utils/metrics.py")
manifest=json.loads(Path("literature/PinPoint_source_manifest.json").read_text())
assert hashlib.sha256(source.read_bytes()).hexdigest()==next(x["sha256"] for x in manifest["files"] if x["path"]=="utils/metrics.py")
ns={"Dict":Dict,"List":List}
exec(compile(ast.Module(body=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef)],type_ignores=[]),str(source),"exec"),ns)
def metric(rank,row):
    ap=ns["calculate_metrics_at_k"](rank,row["positive_set"],10)["ap"]
    return {"ap@10":ap,"precision@1":float(rank[0] in row["positive_set"]),
            "negative_precision@10":sum(s in row["negative_set"] for s in rank[:10])/10}
all_values={};details=[];ties={}
for model in cfg["models"]:
    folder=Path("runs/P031_coral_transfer")/model
    candidates={s:np.load(folder/("candidate_"+s+".npy")) for row in rows for s in row["candidates_to_score"]}
    for condition in cfg["conditions"]:
        for scoring in ["float32_cosine","model_card_bfloat16"]:
            name=model+"/"+condition+"/"+scoring;all_values[name]=[];ties[name]=0
            for row in rows:
                q=np.load(folder/("query_"+row["id"]+"_"+condition+".npy"))
                c=np.stack([candidates[s] for s in row["candidates_to_score"]])
                dtype=torch.float32 if scoring=="float32_cosine" else torch.bfloat16
                qt=F.normalize(torch.tensor(q,dtype=dtype),dim=-1)
                ct=F.normalize(torch.tensor(c,dtype=dtype),dim=-1)
                scores=(ct@qt).float().numpy()
                indices=np.argsort(-scores,kind="stable")
                ranks=[row["candidates_to_score"][i] for i in indices]+row["original_candidates"][20:]
                assert len(ranks)==len(set(ranks))==50
                values=metric(ranks,row);all_values[name].append(values)
                ties[name]+=int(len(set(scores.tolist()))<len(scores))
                details.append({"id":row["id"],"name":name,"ranks":ranks,"scores":scores.tolist(),"metrics":values})
base=json.loads(Path("results/P030_pinpoint_references.json").read_text())
# P030 已将完全相同的参考像素合并；若存在合并，应复用该分组，不能伪称48独立组。
im={x["signature"]:x["pixel_sha256"] for x in json.loads(Path("data/PinPoint/P030_image_manifest.json").read_text())["entries"]}
parent=list(range(len(rows)))
def find(i):
    if parent[i]!=i:parent[i]=find(parent[i])
    return parent[i]
owner={}
for i,row in enumerate(rows):
    for s in row["references"]:
        if im[s] in owner:parent[find(i)]=find(owner[im[s]])
        else:owner[im[s]]=i
groups={}
for i in range(len(rows)):groups.setdefault(find(i),[]).append(i)
blocks=list(groups.values());rng=np.random.default_rng(17031)
indices=[np.concatenate([blocks[i] for i in rng.integers(len(blocks),size=len(blocks))]) for _ in range(5000)]
comparisons=[("coral/dual/float32_cosine","base3b/dual/float32_cosine")]
for model in cfg["models"]:
    comparisons.extend((model+"/dual/float32_cosine",model+"/"+c+"/float32_cosine") for c in ["single","sham","text_only"])
paired={}
for a,b in comparisons:
    paired[a+" minus "+b]={}
    for m in ["ap@10","precision@1","negative_precision@10"]:
        d=np.array([v[m] for v in all_values[a]])-np.array([v[m] for v in all_values[b]])
        paired[a+" minus "+b][m]={"difference_pp":float(d.mean()*100),
            "reference_cluster_95ci_pp":np.percentile([d[ix].mean()*100 for ix in indices],[2.5,97.5]).tolist(),
            "gains":int((d>0).sum()),"losses":int((d<0).sum())}
report={"complete":True,"n_queries":len(rows),"n_reference_rgb_components":len(blocks),
        "conditions":{k:{m:float(np.mean([v[m] for v in a])) for m in a[0]} for k,a in all_values.items()},
        "paired":paired,"ties_anywhere":ties,"P030_comparator":base["conditions"]["canonical_sets"],
        "config_sha256":hashlib.sha256(Path("configs/P031_coral_transfer.json").read_bytes()).hexdigest(),
        "limits":cfg["limits"]}
Path("runs/P031_compared_predictions.json").write_text(json.dumps(details,indent=2)+"\n")
Path("results/P031_coral_transfer.json").write_text(json.dumps(report,indent=2)+"\n")
for k,v in report["conditions"].items():
    if k.endswith("float32_cosine"):print(k,round(v["ap@10"],4),round(v["precision@1"],4))

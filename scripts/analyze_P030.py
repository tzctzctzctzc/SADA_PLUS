"""Reference-grouped comparison with both official-list and coherent-set metrics."""
import ast,collections,hashlib,json,math
from pathlib import Path
from typing import Dict,List
import numpy as np

ROOT=Path("/vepfs/vlms/1");assert Path.cwd().resolve()==ROOT
cfg=json.loads(Path("configs/P030_pinpoint_references.json").read_text())
rawpath=Path("runs/P030_pinpoint_references/predictions.jsonl")
preds=list(map(json.loads,rawpath.read_text().splitlines()))
assert len(preds)==len({r["key"] for r in preds})==cfg["expected_records"]
lookup={(r["id"],r["condition"],r["candidate"]):r for r in preds}
ns={"Dict":Dict,"List":List}
source=Path("third_party/pinpoint-dataset/utils/metrics.py")
sm=json.loads(Path("literature/PinPoint_source_manifest.json").read_text())
assert hashlib.sha256(source.read_bytes()).hexdigest()==next(e["sha256"] for e in sm["files"] if e["path"]=="utils/metrics.py")
nodes=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef)]
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),"exec"),ns)
published=json.loads(Path("data/PinPoint/bge_vl_mllm_s1_retrieval_results_licensed_reranked.json").read_text())
images={r["signature"]:r for r in json.loads(Path("data/PinPoint/P030_image_manifest.json").read_text())["entries"]}
# Refine signature components by exact decoded reference pixels within the pilot.
parent=list(range(len(cfg["records"])))
def find(i):
    if parent[i]!=i:parent[i]=find(parent[i])
    return parent[i]
owners={}
for i,row in enumerate(cfg["records"]):
    for sig in row["references"]:
        px=images[sig]["pixel_sha256"]
        if px in owners:parent[find(i)]=find(owners[px])
        else:owners[px]=i
groups=collections.defaultdict(list)
for i in range(len(parent)):groups[find(i)].append(i)
blocks=list(groups.values());rng=np.random.default_rng(17030)
boot=[np.concatenate([blocks[j] for j in rng.integers(len(blocks),size=len(blocks))]) for _ in range(5000)]
def metrics(rank,row,canonical):
    pos=row["positive_set" if canonical else "positive_original"]
    neg=row["negative_set" if canonical else "negative_original"]
    result={}
    for k in [1,10]:
        m=ns["calculate_metrics_at_k"](rank,pos,k)
        result.update({f"{name}@{k}":value for name,value in m.items()})
    result["NegRecall@10"]=ns["calculate_neg_recall_at_k"](rank,neg,10)
    result["negative_precision@10"]=sum(s in set(neg) for s in rank[:10])/10
    result["unjudged_precision@10"]=sum(s not in set(pos)|set(neg) for s in rank[:10])/10
    no_neg=[s for s in rank if s not in set(neg)]
    result["ap@10_noNeg"]=ns["calculate_metrics_at_k"](no_neg,pos,10)["ap"]
    return result
def sigmoid(m):
    if m>=0:return 1/(1+math.exp(-m))
    e=math.exp(m);return e/(1+e)
names=["original_retrieval","published_reranked","oracle_top20"]+cfg["conditions"]
values={scheme:{name:[] for name in names} for scheme in ["official_lists","canonical_sets"]}
details=[];tie_counts=collections.Counter()
for row in cfg["records"]:
    original=row["original_candidates"]
    candidates=row["candidates_to_score"]
    ranks={"original_retrieval":original,"published_reranked":published[row["id"]]["retrieved_items"],
           "oracle_top20":sorted(candidates,key=lambda s:s in row["positive_set"],reverse=True)+original[20:]}
    scores={}
    for condition in cfg["conditions"]:
        scored=[lookup[row["id"],condition,s] for s in candidates]
        ss=[sigmoid(r["margin"]) for r in scored]
        ranks[condition]=[s for _,s in sorted(zip(ss,candidates),key=lambda p:p[0],reverse=True)]+original[20:]
        assert set(ranks[condition])==set(original) and len(ranks[condition])==50
        tie_counts[condition]+=int(len(set(ss))<len(ss))
        scores[condition]=ss
    result={"id":row["id"],"reference_component":row["reference_component"],"ranks":ranks,"scores":scores,"metrics":{}}
    for scheme in values:
        mm={name:metrics(rank,row,scheme=="canonical_sets") for name,rank in ranks.items()}
        result["metrics"][scheme]=mm
        for name in names:values[scheme][name].append(mm[name])
    details.append(result)
report={"complete":True,"n_queries":len(details),"n_signature_components":cfg["n_selected_components"],
        "n_reference_rgb_components":len(blocks),"conditions":{},"paired":{},"ties_anywhere_in_top20_scores":dict(tie_counts),
        "score_file_sha256":hashlib.sha256(rawpath.read_bytes()).hexdigest(),
        "config_sha256":hashlib.sha256(Path("configs/P030_pinpoint_references.json").read_bytes()).hexdigest(),
        "limits":cfg["limits"]+["Sham assignment couples reference groups; its bootstrap intervals are descriptive, not confirmatory.",
                              "Set-normalized metrics differ from released raw-list denominators and are not directly comparable to paper tables.",
                              "Unjudged images are not automatically treated as explicit negative evidence."]}
for scheme in values:
    report["conditions"][scheme]={name:{k:float(np.mean([v[k] for v in arr])) for k in arr[0]} for name,arr in values[scheme].items()}
    report["paired"][scheme]={}
    for a,b in [("labelled_dual","labelled_single"),("labelled_dual","labelled_sham"),
                ("labelled_single","released_single"),("released_single","original_retrieval")]:
        result={}
        for metric in ["ap@10","precision@1","precision@10","negative_precision@10"]:
            d=np.array([v[metric] for v in values[scheme][a]])-np.array([v[metric] for v in values[scheme][b]])
            result[metric]={"difference_pp":float(d.mean()*100),
                            "reference_cluster_95ci_pp":np.percentile([d[ix].mean()*100 for ix in boot],[2.5,97.5]).tolist(),
                            "query_gains":int((d>0).sum()),"query_losses":int((d<0).sum())}
        report["paired"][scheme][a+" minus "+b]=result
Path("runs/P030_compared_predictions.json").write_text(json.dumps(details,indent=2)+"\n")
Path("results/P030_pinpoint_references.json").write_text(json.dumps(report,indent=2)+"\n")
for name,r in report["conditions"]["canonical_sets"].items():
    print(name,"mAP10",round(r["ap@10"],4),"P1",round(r["precision@1"],4),"negP10",round(r["negative_precision@10"],4))
print("primary",report["paired"]["canonical_sets"]["labelled_dual minus labelled_single"]["ap@10"])

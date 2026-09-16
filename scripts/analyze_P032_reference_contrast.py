"""Preregistered exploratory negative-control calibration on frozen P030 scores."""
import ast, collections, hashlib, json
from pathlib import Path
from typing import Dict,List
import numpy as np
from scipy.stats import rankdata
ROOT=Path("/vepfs/vlms/1");assert Path.cwd().resolve()==ROOT
cfg=json.loads(Path("configs/P032_reference_contrast.json").read_text())
source_config=Path(cfg["source_config"])\nassert hashlib.sha256(source_config.read_bytes()).hexdigest()==cfg["source_config_sha256"]\np030=json.loads(source_config.read_text())
raw=Path("runs/P030_pinpoint_references/predictions.jsonl")
preds=[json.loads(x) for x in raw.read_text().splitlines()]
assert len(preds)==len({x["key"] for x in preds})==p030["expected_records"]
lookup={(x["id"],x["condition"],x["candidate"]):x for x in preds}
source=Path("third_party/pinpoint-dataset/utils/metrics.py")
sm=json.loads(Path("literature/PinPoint_source_manifest.json").read_text())
assert hashlib.sha256(source.read_bytes()).hexdigest()==next(x["sha256"] for x in sm["files"] if x["path"]=="utils/metrics.py")
ns={"Dict":Dict,"List":List}
exec(compile(ast.Module(body=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef)],type_ignores=[]),str(source),"exec"),ns)
images={x["signature"]:x for x in json.loads(Path("data/PinPoint/P030_image_manifest.json").read_text())["entries"]}
parent=list(range(len(p030["records"])))
def find(i):
    if parent[i]!=i:parent[i]=find(parent[i])
    return parent[i]
owner={}
for i,row in enumerate(p030["records"]):
    for sig in row["references"]:
        px=images[sig]["pixel_sha256"]
        if px in owner:parent[find(i)]=find(owner[px])
        else:owner[px]=i
groups=collections.defaultdict(list)
for i in range(len(parent)):groups[find(i)].append(i)
blocks=list(groups.values())
rng=np.random.default_rng(cfg["bootstrap"]["seed"])
boot=[np.concatenate([blocks[j] for j in rng.integers(len(blocks),size=len(blocks))]) for _ in range(cfg["bootstrap"]["replicates"])]
def metrics(rank,row):
    out={}
    for k in [1,10]:
        got=ns["calculate_metrics_at_k"](rank,row["positive_set"],k)
        out.update({f"{name}@{k}":value for name,value in got.items()})
    out["negative_precision@10"]=sum(x in set(row["negative_set"]) for x in rank[:10])/10
    out["unjudged_precision@10"]=sum(x not in set(row["positive_set"])|set(row["negative_set"]) for x in rank[:10])/10
    return out
def auc(pos,neg):
    scores=np.asarray(pos+neg,float);labels=np.asarray([1]*len(pos)+[0]*len(neg))
    ranks=rankdata(scores,method="average")
    return float((ranks[labels==1].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg)))
names=["labelled_dual","labelled_single","labelled_sham","rcr_alpha1","additive_residual","reference_lift"]
values={name:[] for name in names};details=[];ties=collections.Counter();macro_auc=[]
class_lifts={"positive":[],"explicit_negative":[],"unjudged":[]}
for row in p030["records"]:
    candidates=row["candidates_to_score"];original=row["original_candidates"]
    margins={condition:np.array([lookup[row["id"],condition,s]["margin"] for s in candidates],float)
             for condition in ["labelled_dual","labelled_single","labelled_sham"]}
    scores={**margins,
            "rcr_alpha1":2*margins["labelled_dual"]-margins["labelled_sham"],
            "additive_residual":margins["labelled_single"]+margins["labelled_dual"]-margins["labelled_sham"],
            "reference_lift":margins["labelled_dual"]-margins["labelled_sham"]}
    row_metrics={};ranks={}
    for name,score in scores.items():
        order=np.argsort(-score,kind="stable")
        rank=[candidates[i] for i in order]+original[20:]
        assert len(rank)==len(set(rank))==50 and set(rank)==set(original)
        ranks[name]=rank;row_metrics[name]=metrics(rank,row);values[name].append(row_metrics[name])
        ties[name]+=int(len(set(score.tolist()))<len(score))
    pos=[];neg=[]
    for sig,lift in zip(candidates,scores["reference_lift"]):
        if sig in row["positive_set"]:label="positive";pos.append(float(lift))
        elif sig in row["negative_set"]:label="explicit_negative";neg.append(float(lift))
        else:label="unjudged"
        class_lifts[label].append(float(lift))
    macro_auc.append(auc(pos,neg) if pos and neg else None)
    details.append({"id":row["id"],"reference_component":row["reference_component"],
                    "scores":{k:v.tolist() for k,v in scores.items()},"ranks":ranks,"metrics":row_metrics,
                    "mechanism_auc":macro_auc[-1]})
summary={name:{m:float(np.mean([x[m] for x in arr])) for m in arr[0]} for name,arr in values.items()}
paired={}
for a,b in [("rcr_alpha1","labelled_dual"),("additive_residual","labelled_dual"),
            ("labelled_dual","labelled_single"),("labelled_dual","labelled_sham")]:
    result={}
    for metric in ["ap@10","precision@1","precision@10","negative_precision@10"]:
        d=np.array([x[metric] for x in values[a]])-np.array([x[metric] for x in values[b]])
        result[metric]={"difference_pp":float(d.mean()*100),
          "reference_cluster_95ci_pp":np.percentile([d[ix].mean()*100 for ix in boot],[2.5,97.5]).tolist(),
          "query_gains":int((d>0).sum()),"query_losses":int((d<0).sum())}
    paired[a+" minus "+b]=result
eligible=np.array([i for i,x in enumerate(macro_auc) if x is not None])
auc_values=np.array([macro_auc[i] for i in eligible])
auc_boot=[]
for ix in boot:
    selected=[macro_auc[i] for i in ix if macro_auc[i] is not None]
    auc_boot.append(float(np.mean(selected)) if selected else float("nan"))
mechanism={"reference_lift_by_label":{k:{"n":len(v),"mean":float(np.mean(v)) if v else None,
             "median":float(np.median(v)) if v else None} for k,v in class_lifts.items()},
 "macro_query_auc_positive_vs_explicit_negative":{"n_eligible_queries":len(auc_values),
   "mean":float(auc_values.mean()) if len(auc_values) else None,
   "reference_cluster_95ci":np.nanpercentile(auc_boot,[2.5,97.5]).tolist()}}
report={"complete":True,"evidence_status":cfg["evidence_status"],"n_queries":len(details),
 "n_reference_rgb_components":len(blocks),"conditions":summary,"paired":paired,"mechanism":mechanism,
 "ties_anywhere_in_top20_scores":dict(ties),"score_file_sha256":hashlib.sha256(raw.read_bytes()).hexdigest(),
 "config_sha256":hashlib.sha256(Path("configs/P032_reference_contrast.json").read_bytes()).hexdigest(),
 "confirmatory_gate":cfg["confirmatory_gate"],"novelty_boundary":cfg["novelty_boundary"],"limits":cfg["limits"]}
Path("runs/P032_reference_contrast_details.json").write_text(json.dumps(details,indent=2)+"\n")
Path("results/P032_reference_contrast.json").write_text(json.dumps(report,indent=2)+"\n")
for name,x in summary.items():print(name,"mAP10",round(x["ap@10"],5),"P1",round(x["precision@1"],5))
print("PRIMARY",paired["rcr_alpha1 minus labelled_dual"]["ap@10"])
print("MECHANISM",mechanism)

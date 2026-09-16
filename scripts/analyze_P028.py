"""Paired resolution diagnostic, with unchanged source clusters and tie audit."""
import collections, hashlib, json
from pathlib import Path
import numpy as np

ROOT=Path("/vepfs/vlms/1")
assert Path.cwd().resolve()==ROOT
cfg=json.loads(Path("configs/P028_qwen3_resolution.json").read_text())
data_path=Path("runs/P023_data/records.json")
assert hashlib.sha256(data_path.read_bytes()).hexdigest()==cfg["data_sha256"]
data=[r for r in json.loads(data_path.read_text()) if r["split"]=="evaluation"]
assert len(data)==len({r["cluster"] for r in data})==256
ids=[r["id"] for r in data];gold=np.array([r["gold"] for r in data])
old_path=Path("runs/P027_qwen3_imagecode/predictions.jsonl")
new_path=Path("runs/P028_qwen3_resolution/predictions.jsonl")
old=list(map(json.loads,old_path.read_text().splitlines()))
new=list(map(json.loads,new_path.read_text().splitlines()))
assert len(old)==512 and len(new)==256
sets={"official_default":{r["id"]:r for r in old if r["setting"]=="official_default"},
      "bounded_pixels":{r["id"]:r for r in old if r["setting"]=="bounded_pixels"},
      "floor_448":{r["id"]:r for r in new if r["setting"]=="floor_448"}}
assert all(set(v)==set(ids) for v in sets.values())
ann=json.loads(Path("configs/P026_annotation_slices.json").read_text())
slices={"all":np.arange(256)}
for kind in ["video","static"]:
    slices[kind]=np.array([i for i,r in enumerate(data) if r["kind"]==kind])
slices["context_positive"]=np.array([i for i,r in enumerate(data) if r["id"] in ann["positive_slices"]["con"]])
rng=np.random.default_rng(17028)
boots={k:rng.integers(len(ix),size=(5000,len(ix))) for k,ix in slices.items()}
ref={r["id"]:r for r in json.loads(Path("runs/P025_compared_predictions.json").read_text())}
report={"complete":True,"n":256,"n_clusters":256,
        "source_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in [data_path,old_path,new_path]},
        "limits":cfg["limits"],"settings":{},"paired":{}}
vectors={}
for name,records in sets.items():
    gr=[g["image_tokens"] for i in ids for g in records[i]["grids"]]
    item={"tokens":{"min":min(gr),"max":max(gr),"mean":float(np.mean(gr))},
          "scoring_seconds":sum(records[i]["seconds"] for i in ids),"scores":{}}
    for field in ["scores","pre_sigmoid_margins"]:
        vals=np.array([records[i][field] for i in ids])
        pred=vals.argmax(1);top=vals==vals.max(1,keepdims=True)
        expected=top[np.arange(256),gold]/top.sum(1)
        correct=(pred==gold).astype(float)
        vectors[name,field]=(correct,expected)
        # Rank interval avoids treating stable first-index order as semantic rank.
        goldval=vals[np.arange(256),gold,None]
        best_rank=1+(vals>goldval).sum(1)
        worst_rank=(vals>=goldval).sum(1)
        byslice={}
        for sk,ix in slices.items():
            byslice[sk]={"n":len(ix),"correct":int(correct[ix].sum()),
                        "expected_correct_uniform_tie":float(expected[ix].sum()),
                        "top_ties":int((top[ix].sum(1)>1).sum()),
                        "top3_possible":int((best_rank[ix]<=3).sum()),
                        "top3_guaranteed":int((worst_rank[ix]<=3).sum())}
        item["scores"][field]=byslice
    report["settings"][name]=item
for comparator in ["official_default","bounded_pixels"]:
    report["paired"][comparator]={}
    for field in ["scores","pre_sigmoid_margins"]:
        report["paired"][comparator][field]={}
        for metric,j in [("argmax",0),("uniform_tie_expectation",1)]:
            d=vectors["floor_448",field][j]-vectors[comparator,field][j]
            res={}
            for sk,ix in slices.items():
                sub=d[ix]
                res[sk]={"difference_pp":float(sub.mean()*100),
                         "cluster_95ci_pp":np.percentile(sub[boots[sk]].mean(1)*100,[2.5,97.5]).tolist()}
                if j==0:res[sk].update(gains=int((sub>0).sum()),losses=int((sub<0).sum()))
            report["paired"][comparator][field][metric]=res
report["grid_changes"]={}
for comparator in ["official_default","bounded_pixels"]:
    changed=[i for i in ids if sets["floor_448"][i]["grids"]!=sets[comparator][i]["grids"]]
    report["grid_changes"][comparator]={"questions_changed":len(changed),
        "video_changed":sum(r["id"] in changed for r in data if r["kind"]=="video")}
report["context_vs_matching"]={}
for name in sets:
    ix=slices["context_positive"]
    c=vectors[name,"scores"][0]
    base=np.array([ref[i]["predictions"]["matching_score"]==gold[j] for j,i in enumerate(ids)])
    d=(c-base)[ix]
    report["context_vs_matching"][name]={"difference_pp":float(d.mean()*100),
        "cluster_95ci_pp":np.percentile(d[boots["context_positive"]].mean(1)*100,[2.5,97.5]).tolist()}
Path("results/P028_qwen3_resolution.json").write_text(json.dumps(report,indent=2)+"\n")
for name,r in report["settings"].items():
    print(name,{k:v["correct"] for k,v in r["scores"]["scores"].items()})
print("floor - default",report["paired"]["official_default"]["scores"]["argmax"]["all"])

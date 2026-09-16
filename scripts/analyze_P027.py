"""Compare the specialized pointwise reranker with frozen same-pilot baselines."""
from pathlib import Path
import collections,json
import numpy as np
cfg=json.loads(Path("configs/P027_qwen3_imagecode.json").read_text())
data=[r for r in json.loads(Path("runs/P023_data/records.json").read_text()) if r["split"]=="evaluation"]
ref={r["id"]:r for r in json.loads(Path("runs/P025_compared_predictions.json").read_text())}
rows=list(map(json.loads,Path("runs/P027_qwen3_imagecode/predictions.jsonl").read_text().splitlines()))
assert len(rows)==512 and len({(r["setting"],r["id"]) for r in rows})==512
gold=np.array([r["gold"] for r in data])
methods={m:np.array([ref[r["id"]]["predictions"][m] for r in data]) for m in next(iter(ref.values()))["predictions"]}
groups=collections.defaultdict(list)
for i,r in enumerate(data):groups[ref[r["id"]]["cluster"]].append(i)
blocks=list(groups.values());rng=np.random.default_rng(17027)
boot=[np.concatenate([blocks[j] for j in rng.integers(len(blocks),size=len(blocks))]) for _ in range(5000)]
report={"complete":True,"n_evaluation":len(data),"n_clusters":len(blocks),"settings":{},"limits":cfg["limits"]}
for setting in cfg["settings"]:
    byid={r["id"]:r for r in rows if r["setting"]==setting["name"]}
    assert set(byid)==set(ref)
    res={"scores":{},"token_range":[min(g["image_tokens"] for r in byid.values() for g in r["grids"]),
                                   max(g["image_tokens"] for r in byid.values() for g in r["grids"])]}
    for field in ["scores","pre_sigmoid_margins"]:
        values=np.array([byid[r["id"]][field] for r in data]);pred=values.argmax(1);c=(pred==gold).astype(int)
        tied=values==values.max(1,keepdims=True)
        result={"correct":int(c.sum()),"n_top_ties":int((tied.sum(1)>1).sum()),
                "expected_correct_uniform_tie":float((tied[np.arange(len(data)),gold]/tied.sum(1)).sum()),
                "by_kind":{kind:{"n":sum(r["kind"]==kind for r in data),
                                 "correct":int(sum(c[i] for i,r in enumerate(data) if r["kind"]==kind))} for kind in ["static","video"]},
                "comparisons":{}}
        for baseline in ["raw_logits","patch_attention_calibrated","patch_density_calibrated","caption_logprob_mean","matching_score","SigLIP2"]:
            d=c-(methods[baseline]==gold).astype(int)
            result["comparisons"][baseline]={"difference_pp":float(d.mean()*100),
                "cluster_95ci_pp":np.percentile([d[ix].mean()*100 for ix in boot],[2.5,97.5]).tolist(),
                "gains":int((d>0).sum()),"losses":int((d<0).sum())}
        res["scores"][field]=result
    report["settings"][setting["name"]]=res
Path("results/P027_qwen3_imagecode.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))

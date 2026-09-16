"""Compare frozen P023/P025 predictions on explicitly positive author annotations."""
import collections, hashlib, json
from pathlib import Path
import numpy as np

cfg = json.loads(Path("configs/P026_annotation_slices.json").read_text())
assert hashlib.sha256(Path("configs/P023_imagecode.json").read_bytes()).hexdigest() == cfg["P023_config_sha256"]
pred_path = Path("runs/P025_compared_predictions.json")
predictions = {r["id"]: r for r in json.loads(pred_path.read_text())}
assert set(predictions) == {r["id"] for r in cfg["records"]}
out = {
    "complete": True, "n_evaluation": len(predictions),
    "source_sha256": cfg["source_sha256"],
    "prediction_sha256": hashlib.sha256(pred_path.read_bytes()).hexdigest(),
    "status_counts": cfg["status_counts"], "slices": {},
    "limitations": cfg["limitations"],
}
for code, ids in cfg["positive_slices"].items():
    rows = [predictions[i] for i in ids]
    correct = {m:np.array([r["predictions"][m] == r["gold"] for r in rows], dtype=int)
               for m in rows[0]["predictions"]}
    groups = collections.defaultdict(list)
    for i,r in enumerate(rows):
        groups[r["cluster"]].append(i)
    blocks = list(groups.values())
    rng = np.random.default_rng(cfg["bootstrap"]["seed"])
    draws = [np.concatenate([blocks[j] for j in rng.integers(len(blocks), size=len(blocks))])
             for _ in range(cfg["bootstrap"]["replicates"])]
    result = {"n": len(rows), "n_clusters": len(blocks),
              "correct": {m:int(c.sum()) for m,c in correct.items()}, "comparisons": {}}
    for a,b in cfg["comparisons"]:
        d = correct[a] - correct[b]
        result["comparisons"][a+" minus "+b] = {
            "difference_pp": float(d.mean()*100),
            "cluster_95ci_pp": np.percentile([d[ix].mean()*100 for ix in draws], [2.5,97.5]).tolist(),
            "gains": int((d>0).sum()), "losses": int((d<0).sum())
        }
    out["slices"][code] = result
Path("results/P026_annotation_slices.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))

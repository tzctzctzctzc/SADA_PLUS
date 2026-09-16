import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import binomtest

root=Path("/vepfs/vlms/1")
paths={m:root/f"runs/P002_mantis_{m}" for m in ("baseline","scale")}
assert all(json.loads((p/"summary.json").read_text())["complete"] for p in paths.values())
pred={m:{r["id"]:r for r in [json.loads(line) for line in (p/"predictions.jsonl").read_text().splitlines()]} for m,p in paths.items()}
assert set(pred["baseline"])==set(pred["scale"])
ids=list(pred["baseline"])
a=np.array([pred["baseline"][i]["correct"] for i in ids],dtype=int)
b=np.array([pred["scale"][i]["correct"] for i in ids],dtype=int)
gain=int(((a==0)&(b==1)).sum());loss=int(((a==1)&(b==0)).sum())
rng=np.random.default_rng(20260917)
boot=(b-a)[rng.integers(0,len(ids),size=(20000,len(ids)))].mean(1)
report={"n":len(ids),"baseline_correct":int(a.sum()),"scale_correct":int(b.sum()),
    "baseline_accuracy":float(a.mean()),"scale_accuracy":float(b.mean()),
    "delta_percentage_points":float(100*(b-a).mean()),"gains":gain,"losses":loss,
    "mcnemar_exact_two_sided_p":float(binomtest(gain,gain+loss,0.5).pvalue) if gain+loss else 1.0,
    "paired_question_bootstrap_95ci_percentage_points":(100*np.quantile(boot,[0.025,0.975])).tolist(),
    "limitations":["Exploratory question-level interval; shared images may create additional dependence",
        "Fixed 3B example scaling parameters transferred to 7B, not audited 7B optimum",
        "One sampling seed; Mantis already used for mechanism exploration"],
    "changed_ids":[{"id":i,"category":pred["baseline"][i]["category"],"baseline":pred["baseline"][i]["correct"],"scale":pred["scale"][i]["correct"]} for i in ids if pred["baseline"][i]["correct"]!=pred["scale"][i]["correct"]]}
for field in ("question_type","category"):
    groups=defaultdict(list)
    for j,i in enumerate(ids):
        groups[pred["baseline"][i][field]].append(j)
    report[field]={g:{"n":len(js),"baseline_correct":int(a[js].sum()),"scale_correct":int(b[js].sum())} for g,js in groups.items()}
report["tokens"]={m:{"hit_limit":sum(r["hit_token_limit"] for r in p.values()),
    "seconds":sum(r["seconds"] for r in p.values()),"peak_memory_gb":max(r["peak_memory_gb"] for r in p.values())} for m,p in pred.items()}
(root/"runs/P002_paired_analysis.json").write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ("category","changed_ids")},indent=2))

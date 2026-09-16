import json,re
from pathlib import Path
from collections import Counter
from datasets import load_dataset
from scipy.stats import binomtest
root=Path("/vepfs/vlms/1")
ds=load_dataset("parquet",data_files={"test":[str(p) for p in (root/"data/Mantis-Eval").rglob("*.parquet")]},split="test").remove_columns("images")
preds={m:{r["id"]:r for r in map(json.loads,(root/f"runs/P002_mantis_{m}/predictions.jsonl").read_text().splitlines())} for m in ("baseline","scale")}
rows=[]
for d in ds:
    if d["question_type"]!="multi-choice":continue
    maps=[re.findall(r"image\s*(\d+)",o,re.I) for o in d["options"]]
    if not maps or not all(len(x)==1 for x in maps):continue
    indices=[int(x[0]) for x in maps]
    n=preds["baseline"][d["id"]]["images"]
    if sorted(indices)!=list(range(1,n+1)):continue
    r={"id":d["id"],"question":d["question"],"n_images":n}
    for m in ("baseline","scale"):
        p=preds[m][d["id"]]
        letter=p["parsed_prediction"]
        r[m+"_index"]=indices[ord(letter)-65] if isinstance(letter,str) and len(letter)==1 and 0<=ord(letter)-65<len(indices) else None
        r[m+"_correct"]=p["correct"]
    letter=d["answer"].strip("() \n")
    r["gold_index"]=indices[ord(letter)-65]
    rows.append(r)
valid=[r for r in rows if r["baseline_index"] is not None and r["scale_index"] is not None]
lower=sum(r["scale_index"]<r["baseline_index"] for r in valid)
higher=sum(r["scale_index"]>r["baseline_index"] for r in valid)
out={"n":len(rows),"n_valid":len(valid),"shift_to_lower_index":lower,"shift_to_higher_index":higher,
    "exploratory_binomial_p":float(binomtest(lower,lower+higher,0.5).pvalue) if lower+higher else 1,
    "accuracy":{m:sum(r[m+"_correct"] for r in rows)/len(rows) for m in ("baseline","scale")},
    "prediction_counts":{m:dict(Counter(r[m+"_index"] for r in valid)) for m in ("baseline","scale")},
    "gold_counts":dict(Counter(r["gold_index"] for r in valid)),
    "limitations":["Source image order is confounded with option letter/order in these questions",
        "Repeated/shared images and post-hoc subset definition; exploratory p value is not confirmatory",
        "Need factorial image permutation and option permutation to distinguish source-index bias from answer-letter bias"],
    "rows":rows}
(root/"runs/P002_source_index_analysis.json").write_text(json.dumps(out,indent=2))
print(json.dumps({k:v for k,v in out.items() if k!="rows"},indent=2))

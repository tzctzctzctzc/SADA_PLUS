"""Recompute published rankings with pinned metrics; keep label-policy scopes explicit."""
import hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path("/vepfs/vlms/1");assert Path.cwd().resolve()==ROOT
meta=json.loads(Path("literature/PinPoint_source_manifest.json").read_text())
vendor=ROOT/"third_party/pinpoint-dataset"
for file in meta["files"]:
    assert hashlib.sha256((vendor/file["path"]).read_bytes()).hexdigest()==file["sha256"]
sys.path.insert(0,str(vendor))
spec=importlib.util.spec_from_file_location("p030_official_metrics",vendor/"evaluate.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
original=pd.read_parquet("data/PinPoint/pinpoint_licensed.parquet")
canonical=original.copy()
for column in ["positive_candidates","negative_candidates"]:
    canonical[column]=canonical[column].map(lambda x:list(dict.fromkeys(x.tolist())))
consistent=canonical[[not set(r.positive_candidates)&set(r.negative_candidates) for _,r in canonical.iterrows()]].copy()
manifest=json.loads(Path("literature/PinPoint_candidates_manifest.json").read_text())
report={"complete":True,"revision":meta["revision"],
        "row_counts":{"official_lists":len(original),"canonical_all":len(canonical),"canonical_consistent":len(consistent)},
        "results":{},"limits":["Published rankings only; this does not rerun models.",
             "canonical_all still includes positive-negative conflicts; only duplicate IDs are removed.",
             "canonical_consistent changes both annotation policy and query cohort and is not directly comparable to paper/README numbers.",
             "Image licensing/remapping as a cause of duplicates or conflicts has not been established."]}
for entry in manifest["files"]:
    p=ROOT/entry["local_path"];assert hashlib.sha256(p.read_bytes()).hexdigest()==entry["sha256"]
    ranks=json.loads(p.read_text())
    label="published_reranked" if "_reranked" in p.name else "published_original"
    report["results"][label]={}
    for scheme,df in [("official_lists",original),("canonical_all",canonical),("canonical_consistent",consistent)]:
        result=module.evaluate_model(label+"_"+scheme,ranks,df)
        assert result["total_queries"]==len(df)
        report["results"][label][scheme]=result
    decreases=[]
    for _,r in original.iterrows():
        rank=ranks[module.normalize_query_id(r.query_id)]["retrieved_items"]
        pos=r.positive_candidates.tolist();neg=r.negative_candidates.tolist()
        before=module.calculate_metrics_at_k(rank,pos,10)["ap"]
        after=module.calculate_metrics_at_k(module.filter_out_negatives(rank,neg),pos,10)["ap"]
        if after+1e-12<before:decreases.append(module.normalize_query_id(r.query_id))
    report["results"][label]["queries_where_removing_negatives_reduces_AP10"]=decreases
ceilings=[]
for _,r in original.iterrows():
    n=len(r.positive_candidates);u=len(set(r.positive_candidates))
    ceilings.append(min(u,10)/min(n,10) if n else 0.)
report["raw_list_AP10_upper_bound_due_to_duplicate_labels"]={
    "n_queries_with_positive_labels":int(sum(len(x)>0 for x in original.positive_candidates)),
    "n_queries_below_one":int(sum(c<1 for c in ceilings)),
    "mean":float(np.mean(ceilings)),
    "definition":"With unique retrieved images, even a perfect ordering of all unique positives has AP@10 <= min(unique positives,10)/min(original positive list length,10)."}
Path("results/PinPoint_metric_replay.json").write_text(json.dumps(report,indent=2,default=float)+"\n")
for name,data in report["results"].items():
    print(name,{scheme:round(data[scheme]["mAP@10"],5) for scheme in report["row_counts"]})
print("duplicate-label upper bound",report["raw_list_AP10_upper_bound_due_to_duplicate_labels"])

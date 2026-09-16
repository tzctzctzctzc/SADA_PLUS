"""Freeze a reference-grouped pilot before new model scoring."""
import collections,datetime,hashlib,json,re
from pathlib import Path
import pandas as pd

ROOT=Path("/vepfs/vlms/1");assert Path.cwd().resolve()==ROOT
seed="P030-pinpoint-v1"
digest=lambda s:hashlib.sha256(s.encode()).hexdigest()
df=pd.read_parquet("data/PinPoint/pinpoint_licensed.parquet")
meta=json.loads(Path("literature/PinPoint_source_manifest.json").read_text())
candidate_meta=json.loads(Path("literature/PinPoint_candidates_manifest.json").read_text())
rawpath=Path("data/PinPoint/bge_vl_mllm_s1_retrieval_results_licensed.json")
raw=json.loads(rawpath.read_text())
qid=lambda x:str(x).removeprefix("query_").zfill(5)
assert set(raw)==set(df.query_id.map(qid))
assert all(len(v["retrieved_items"])==len(set(v["retrieved_items"]))==50 for v in raw.values())
valid_sig=lambda x:isinstance(x,str) and re.fullmatch(r"[0-9a-f]{32}",x) is not None
multi=df[df.query_image_signature2.map(valid_sig)]
parent={}
def find(x):
    parent.setdefault(x,x)
    if parent[x]!=x:parent[x]=find(parent[x])
    return parent[x]
for _,r in multi.iterrows():
    a,b=find(r.query_image_signature),find(r.query_image_signature2)
    parent[max(a,b)]=min(a,b)
groups=collections.defaultdict(list)
for _,r in multi.iterrows():groups[find(r.query_image_signature)].append(r)
conflicts=[qid(r.query_id) for _,r in df.iterrows()
           if set(r.positive_candidates)&set(r.negative_candidates)]
eligible={}
for group,rows in groups.items():
    clean=[r for r in rows if qid(r.query_id) not in conflicts]
    if clean:eligible[group]=clean
selected_groups=sorted(eligible,key=lambda g:digest(seed+"|group|"+g))[:48]
assert len(selected_groups)==48
records=[]
for group in selected_groups:
    r=min(eligible[group],key=lambda r:digest(seed+"|query|"+qid(r.query_id)))
    q=qid(r.query_id);pos=list(r.positive_candidates);neg=list(r.negative_candidates)
    assert not set(pos)&set(neg)
    records.append({"id":q,"source_query_id":r.query_id,"reference_component":group,
                    "references":[r.query_image_signature,r.query_image_signature2],
                    "instruction":r.instruction,"positive_original":pos,"negative_original":neg,
                    "positive_set":sorted(set(pos)),"negative_set":sorted(set(neg)),
                    "original_candidates":raw[q]["retrieved_items"],
                    "candidates_to_score":raw[q]["retrieved_items"][:20]})
for i,r in enumerate(records):
    r["sham_reference2"]=records[(i+1)%len(records)]["references"][1]
    assert r["sham_reference2"] not in r["references"]
all_sigs=sorted({sig for r in records for sig in r["references"]+r["candidates_to_score"]})
assert all(valid_sig(x) for x in all_sigs)
cfg={"id":"P030","seed":seed,"purpose":"Complete-input reproduction/control of the released pointwise reranker on two-reference queries; protocol correction is not a new method.",
     "created_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),
     "source_revision":meta["revision"],"model":"Qwen2.5-VL-7B-Instruct",
     "model_revision":"cc594898137f460bfe9f0759e9844b3ce807cfb5",
     "gpu_uuid":"GPU-7ce1c8eb-f665-99cf-2541-3c905a3f98f1",
     "backend":"sdpa","dtype":"bfloat16","min_pixels":3136,"max_pixels":1048576,
     "batch_size":4,"logits_to_keep":1,
     "conditions":["released_single","labelled_single","labelled_dual","labelled_sham"],
     "prompt_protocol":"Pinned official SYSTEM_PROMPT/BASE_INSTRUCTIONS and build_messages. Labelled versions mark the first reference as <img1> and the candidate explicitly; dual adds reference <img2>; sham replaces only the second image by a cyclic permutation across disjoint reference groups. All other text is identical.",
     "n_reference_components":len(groups),"n_eligible_components":len(eligible),
     "n_selected_components":48,"expected_records":48*20*4,
     "selection":"One seed-hash query per seed-hash-selected reference connected component, excluding rows with contradictory positive/negative labels before any new GPU scores. No selection for candidate recall.",
     "annotation_policy":"Keep original annotation lists for exact official metrics. Primary coherent-set metrics deduplicate positive and negative IDs; conflicting rows are excluded only from this fixed diagnostic cohort and counted explicitly.",
     "primary_comparison":"labelled_dual minus labelled_single on set-normalized mAP@10",
     "secondary_comparisons":["labelled_dual minus labelled_sham","labelled_single minus released_single","released_single minus original retrieval"],
     "other_metrics":["set-normalized precision@1","set-normalized precision@10","explicit negative precision@10","official raw-list metrics","top20 oracle candidate ceiling","stable-score ties"],
     "ranking":"Float64 sigmoid of raw True-minus-False logits; descending stable sort preserves original candidate order in ties; untouched ranks 21-50 appended.",
     "bootstrap":{"samples":5000,"unit":"shared-reference connected component","seed":17030},
     "reserved_reference_components":sorted(set(groups)-set(selected_groups)),
     "reserved_query_ids":[qid(r.query_id) for g,rs in sorted(groups.items()) if g not in selected_groups for r in rs],
     "new_model_scores_on_reserved_groups_seen":False,
     "image_signatures":all_sigs,
     "source_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in [rawpath,Path("data/PinPoint/pinpoint_licensed.parquet")]},
     "limits":["48-query diagnostic pilot, not full PinPoint or SOTA.",
               "Reference groups are disjoint by signature; retrieval candidates can be shared across groups.",
               "The 1M upper pixel budget is explicit; released API serving defaults are unspecified.",
               "Published candidate lists are reused; this is not a reproduction of the first-stage retriever.",
               "Sham references are input corruptions under the original labels, not counterfactual queries with relabelled ground truth."],
     "records":records}
out=Path("configs/P030_pinpoint_references.json")
assert not out.exists()
out.write_text(json.dumps(cfg,indent=2)+"\n")
audit={"n_rows":len(df),"n_multi_rows":len(multi),"n_conflicting_all":len(conflicts),
       "n_conflicting_multi":sum(qid(r.query_id) in conflicts for _,r in multi.iterrows()),
       "positive_duplicate_rows":sum(len(r.positive_candidates)!=len(set(r.positive_candidates)) for _,r in df.iterrows()),
       "negative_duplicate_rows":sum(len(r.negative_candidates)!=len(set(r.negative_candidates)) for _,r in df.iterrows()),
       "conflicting_query_ids":conflicts,"reference_components":len(groups),
       "eligible_reference_components":len(eligible),"selected_components":48,
       "reserved_components":len(cfg["reserved_reference_components"]),
       "selected_queries_with_positive_duplicates":sum(len(r["positive_original"])!=len(r["positive_set"]) for r in records),
       "n_images_to_fetch":len(all_sigs),"config_sha256":hashlib.sha256(out.read_bytes()).hexdigest(),
       "cohort_exclusion_is_pre_model":True}
Path("results/P030_metadata_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
print({k:v for k,v in audit.items() if k!="conflicting_query_ids"})

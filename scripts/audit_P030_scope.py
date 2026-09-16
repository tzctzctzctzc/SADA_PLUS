"""Audit category/domain coverage of the frozen P030 cohort without model scores."""
import collections,hashlib,json,re
from pathlib import Path
import pandas as pd
ROOT=Path("/vepfs/vlms/1");assert Path.cwd().resolve()==ROOT
cfg_path=Path("configs/P030_pinpoint_references.json")
cfg=json.loads(cfg_path.read_text())
df=pd.read_parquet("data/PinPoint/pinpoint_licensed.parquet")
valid=lambda x:isinstance(x,str) and re.fullmatch(r"[0-9a-f]{32}",x) is not None
multi=df[df.query_image_signature2.map(valid)].copy()
qid=lambda x:str(x).removeprefix("query_").zfill(5)
multi["id"]=multi.query_id.map(qid)
multi["conflict"]=[bool(set(x.positive_candidates)&set(x.negative_candidates)) for _,x in multi.iterrows()]
parent={}
def find(x):
    parent.setdefault(x,x)
    if parent[x]!=x:parent[x]=find(parent[x])
    return parent[x]
for _,row in multi.iterrows():
    a,b=find(row.query_image_signature),find(row.query_image_signature2)
    parent[max(a,b)]=min(a,b)
multi["reference_component"]=[find(x) for x in multi.query_image_signature]
selected={x["id"] for x in cfg["records"]}
reserved=set(cfg["reserved_query_ids"])
assert selected.isdisjoint(reserved)
def count(frame,column):
    return dict(sorted(collections.Counter(map(str,frame[column])).items()))
def scope(frame):
    clean=frame[~frame.conflict]
    return {"n_rows":len(frame),"n_clean_rows":len(clean),
      "n_reference_components":frame.reference_component.nunique(),
      "query_category_all":count(frame,"query_category"),
      "query_category_clean":count(clean,"query_category"),
      "l1_interest_clean":count(clean,"l1_interest"),
      "length_category_clean":count(clean,"length_category")}
pilot=multi[multi.id.isin(selected)]
reserved_rows=multi[multi.id.isin(reserved)]
assert len(pilot)==len(selected)
assert set(reserved_rows.reference_component)==set(cfg["reserved_reference_components"])
report={"complete":True,"config_sha256":hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
 "all_multi":scope(multi),"pilot_selected_queries":scope(pilot),
 "reserved_components_all_rows":scope(reserved_rows),
 "non_composition_clean_query_ids":multi[(~multi.conflict)&(multi.query_category!="Composition")].id.tolist(),
 "limits":["The pilot was selected by component/query hashes rather than category stratification.",
 "Rows in one reference component are not independent and can have different query categories.",
 "The category audit is metadata-only and makes no claim about model performance."]}
Path("results/P030_scope_audit.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({"all_clean":report["all_multi"]["query_category_clean"],
 "pilot":report["pilot_selected_queries"]["query_category_clean"],
 "reserved":report["reserved_components_all_rows"]["query_category_clean"],
 "non_composition_clean":len(report["non_composition_clean_query_ids"])},indent=2))

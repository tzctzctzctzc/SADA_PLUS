"""Trace pinned released reranker messages with real two-reference metadata."""
import ast, hashlib, json, re
from pathlib import Path
from typing import Dict,List,Optional,Tuple
import pandas as pd

ROOT=Path("/vepfs/vlms/1")
assert Path.cwd().resolve()==ROOT
meta=json.loads(Path("literature/PinPoint_source_manifest.json").read_text())
source=Path("third_party/pinpoint-dataset/rerank.py")
expected=next(f for f in meta["files"] if f["path"]=="rerank.py")
assert hashlib.sha256(source.read_bytes()).hexdigest()==expected["sha256"]
namespace={"Dict":Dict,"List":List,"Tuple":Tuple,"Optional":Optional,
           "pd":pd,"VLMClient":object,"tqdm":lambda x,**kwargs:x}
normalize=next(n for n in ast.parse(Path("third_party/pinpoint-dataset/utils/data_utils.py").read_text()).body
               if isinstance(n,ast.FunctionDef) and n.name=="normalize_query_id")
exec(compile(ast.Module(body=[normalize],type_ignores=[]),"<normalize_query_id>","exec"),namespace)
tree=ast.parse(source.read_text())
wanted={"signature_to_url","build_messages","rerank_query","rerank_results"}
nodes=[n for n in tree.body if
       (isinstance(n,ast.FunctionDef) and n.name in wanted) or
       (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in
                                       {"SYSTEM_PROMPT","BASE_INSTRUCTIONS"} for t in n.targets))]
assert {n.name for n in nodes if isinstance(n,ast.FunctionDef)}==wanted
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),"exec"),namespace)
df=pd.read_parquet("data/PinPoint/pinpoint_licensed.parquet")
valid2=df["query_image_signature2"].map(lambda x:isinstance(x,str) and re.fullmatch(r"[0-9a-f]{32}",x) is not None)
multi=df[valid2].copy()
assert len(multi)>0
results={}
for _,row in multi.iterrows():
    candidates=list(row["positive_candidates"])
    assert candidates
    qid=namespace["normalize_query_id"](row["query_id"])
    assert qid not in results
    results[qid]={"retrieved_items":[candidates[0]]}
class Capture:
    def __init__(self):self.payloads=[]
    def score_candidate(self,messages):
        self.payloads.append(messages)
        return 0.5  # No model/API/image access; test only the input information path.

with_second=Capture();without_second=Capture()
namespace["rerank_results"](with_second,results,multi,top_n=1)
namespace["rerank_results"](without_second,results,multi.drop(columns=["query_image_signature2"]),top_n=1)
assert len(with_second.payloads)==len(without_second.payloads)==len(multi)
traces=[]
for (_,row),a,b in zip(multi.iterrows(),with_second.payloads,without_second.payloads):
    ja=json.dumps(a,sort_keys=True)
    urls=[c["image_url"]["url"] for m in a for c in m.get("content",[])
          if isinstance(c,dict) and c.get("type")=="image_url"]
    traces.append({"query_id":row["query_id"],"image_count":len(urls),
                   "first_present":namespace["signature_to_url"](row["query_image_signature"]) in urls,
                   "second_present":namespace["signature_to_url"](row["query_image_signature2"]) in urls,
                   "identical_without_second_column":a==b,
                   "payload_sha256":hashlib.sha256(ja.encode()).hexdigest()})
Path("runs/PinPoint_reference_trace.json").write_text(json.dumps(traces,indent=2)+"\n")
report={"complete":True,"repo":meta["repo"],"revision":meta["revision"],
        "source_sha256":expected["sha256"],"n_rows":len(df),
        "n_unique_query_ids":int(df.query_id.nunique()),
        "n_valid_second_reference_rows":len(multi),
        "n_distinct_ordered_reference_pairs":len(multi[["query_image_signature","query_image_signature2"]].drop_duplicates()),
        "n_traced":len(traces),
        "payload_image_counts":pd.Series([t["image_count"] for t in traces]).value_counts().to_dict(),
        "first_reference_present":sum(t["first_present"] for t in traces),
        "second_reference_present":sum(t["second_present"] for t in traces),
        "identical_after_dropping_second_column":sum(t["identical_without_second_column"] for t in traces),
        "source_locations":{n.name:{"start":n.lineno,"end":n.end_lineno}
                            for n in nodes if isinstance(n,ast.FunctionDef)},
        "scope":"Pinned publicly released rerank.py input path only. No GPU, API calls, image downloads or model scoring.",
        "limits":["This does not establish which code/version produced the paper tables.",
                  "A missing second reference does not quantify accuracy impact; a complete-input baseline is still required.",
                  "The literal string None is present in metadata; valid second references are identified by the 32-hex signature format.",
                  "Multiple instruction variants reuse reference pairs; future splits must account for shared images."]}
Path("results/PinPoint_reference_audit.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))

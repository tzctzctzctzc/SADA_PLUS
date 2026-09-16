import hashlib,json
from collections import Counter
from pathlib import Path
import pyarrow.parquet as pq
root=Path("/vepfs/vlms/1")
files=sorted((root/"data/MUIRBENCH").rglob("*.parquet"))
assert len(files)==5
parent={}; image_owner={}; metadata={}
def find(x):
    parent.setdefault(x,x)
    if parent[x]!=x: parent[x]=find(parent[x])
    return parent[x]
def union(a,b):
    a,b=find(a),find(b)
    if a!=b: parent[max(a,b)]=min(a,b)
for file in files:
    rows=pq.read_table(file,columns=["idx","counterpart_idx","task","image_list"]).to_pylist()
    for r in rows:
        idx=r["idx"];metadata[idx]={k:r[k] for k in ("counterpart_idx","task")}
        find(idx)
        if r["counterpart_idx"]: union(idx,r["counterpart_idx"])
        for image in r["image_list"]:
            assert image["bytes"] is not None
            digest=hashlib.sha256(image["bytes"]).hexdigest()
            if digest in image_owner: union(idx,image_owner[digest])
            else: image_owner[digest]=idx
groups={}
for idx in metadata: groups.setdefault(find(idx),[]).append(idx)
seed="gpu3-muir-v1-20260917"
dev=[];test=[]
for key,ids in sorted(groups.items()):
    fraction=int(hashlib.sha256((seed+":"+key).encode()).hexdigest()[:16],16)/2**64
    (dev if fraction<0.1 else test).extend(ids)
report={"seed":seed,"dataset_revision":json.loads((root/"data/MUIRBENCH/source_manifest.json").read_text())["revision"],
    "grouping":"Connected components of counterpart pairs and exact SHA256 image-byte duplicates. No questions, options, answers, or model predictions accessed.",
    "dev_ids":sorted(dev),"test_ids":sorted(test),"n":len(metadata),"n_groups":len(groups),
    "largest_groups":sorted(map(len,groups.values()),reverse=True)[:10],
    "dev_tasks":dict(Counter(metadata[i]["task"] for i in dev)),
    "test_tasks":dict(Counter(metadata[i]["task"] for i in test))}
(root/"configs/muir_split_v1.json").write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ("dev_ids","test_ids")},indent=2))
print("dev",len(dev),"test",len(test))

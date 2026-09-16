import concurrent.futures,json,urllib.request
from pathlib import Path
root=Path("/vepfs/vlms/1/third_party/LAD")
root.mkdir(parents=True,exist_ok=True)
def get(url):
    req=urllib.request.Request(url,headers={"User-Agent":"gpu3-research"})
    return urllib.request.urlopen(req,timeout=30).read()
tree=json.loads(get("https://api.github.com/repos/brightXian/LAD/git/trees/main?recursive=1"))
(root/"source_manifest.json").write_text(json.dumps(tree,indent=2))
files=[x for x in tree["tree"] if x["type"]=="blob" and x["path"].endswith((".py",".toml",".md",".sh"))]
def fetch(entry):
    path=entry["path"];dest=(root/path).resolve()
    assert dest.is_relative_to(root)
    data=get("https://raw.githubusercontent.com/brightXian/LAD/"+tree["sha"]+"/"+path)
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
    print(path,len(data),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(fetch,files))
print("DONE",tree["sha"],len(files),flush=True)

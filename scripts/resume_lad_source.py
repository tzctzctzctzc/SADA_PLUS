import base64,concurrent.futures,hashlib,json,time,urllib.request
from pathlib import Path
root=Path("/vepfs/vlms/1/third_party/LAD")
tree=json.loads((root/"source_manifest.json").read_text())
def blob_hash(data): return hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()
files=[e for e in tree["tree"] if e["type"]=="blob" and e["path"].endswith((".py",".toml",".md",".sh"))]
def fetch(e):
    p=(root/e["path"]).resolve()
    assert p.is_relative_to(root)
    if p.exists() and blob_hash(p.read_bytes())==e["sha"]:
        return
    for attempt in range(3):
        try:
            url="https://api.github.com/repos/brightXian/LAD/git/blobs/"+e["sha"]
            req=urllib.request.Request(url,headers={"User-Agent":"gpu3-research"})
            data=json.load(urllib.request.urlopen(req,timeout=25))
            raw=base64.b64decode(data["content"])
            assert blob_hash(raw)==e["sha"]
            p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
            print("OK",e["path"],flush=True);return
        except Exception as error:
            print("RETRY",e["path"],attempt+1,type(error).__name__,flush=True)
            if attempt==2:raise
            time.sleep(2)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(fetch,files))
print("DONE",tree["sha"],len(files),flush=True)

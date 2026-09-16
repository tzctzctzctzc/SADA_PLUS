"""Fetch only the preregistered PinPoint pilot images and preserve hashes."""
import concurrent.futures,hashlib,io,json,pathlib,sys,time
import requests
from PIL import Image
ROOT=pathlib.Path("/vepfs/vlms/1");assert pathlib.Path.cwd().resolve()==ROOT
cfg_path=ROOT/"configs/P030_pinpoint_references.json"
cfg=json.loads(cfg_path.read_text())
directory=ROOT/"data/PinPoint/images";directory.mkdir(parents=True,exist_ok=True)
sigs=cfg["image_signatures"]
def inspect(sig,b):
    with Image.open(io.BytesIO(b)) as im:
        im.load();fmt=im.format;im=im.convert("RGB")
        assert min(im.size)>0
        pixel_hash=hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
        return {"signature":sig,"path":str((directory/(sig+".jpg")).relative_to(ROOT)),
                "bytes":len(b),"sha256":hashlib.sha256(b).hexdigest(),
                "pixel_sha256":pixel_hash,"size":list(im.size),"format":fmt}
def fetch(sig):
    url=f"https://i.pinimg.com/736x/{sig[:2]}/{sig[2:4]}/{sig[4:6]}/{sig}.jpg"
    p=directory/(sig+".jpg")
    if p.exists():
        try:return dict(inspect(sig,p.read_bytes()),url=url,cached=True)
        except Exception:pass
    error=None
    for attempt in range(3):
        try:
            rr=requests.get(url,timeout=(10,20));rr.raise_for_status()
            entry=inspect(sig,rr.content)
            partial=p.with_suffix(".part");partial.write_bytes(rr.content);partial.replace(p)
            return dict(entry,url=url,cached=False)
        except Exception as ex:
            error=str(ex)
            if attempt<2:time.sleep(1+attempt)
    return {"signature":sig,"url":url,"error":error}
entries=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
    futures={ex.submit(fetch,sig):sig for sig in sigs}
    with (ROOT/"runs/P030_image_fetch.jsonl").open("w") as journal:
        for fut in concurrent.futures.as_completed(futures):
            r=fut.result();entries.append(r)
            journal.write(json.dumps(r)+"\n");journal.flush()
            if len(entries)%50==0:print(len(entries),"/",len(sigs),"failed",sum("error" in x for x in entries),flush=True)
entries.sort(key=lambda e:e["signature"])
failures=[r for r in entries if "error" in r]
report={"complete":not failures,"config_sha256":hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
        "n_expected":len(sigs),"n_success":len(entries)-len(failures),"n_failed":len(failures),"entries":entries}
(ROOT/"data/PinPoint/P030_image_manifest.json").write_text(json.dumps(report,indent=2)+"\n")
print("complete",report["complete"],"success",report["n_success"],"failed",report["n_failed"])
sys.exit(0 if report["complete"] else 1)

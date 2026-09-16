"""Replace only the calibration item with verified pixel overlap, before any evaluation inference."""
import hashlib, io, json, zipfile
from pathlib import Path
from PIL import Image

root=Path("/vepfs/vlms/1")
cfg_path=root/"configs/P023_imagecode.json"
cfg=json.loads(cfg_path.read_text())
assert "data_integrity_amendment" not in cfg
assert not (root/"runs/P023_imagecode_vlm/predictions.jsonl").exists()
original_cfg_sha=hashlib.sha256(cfg_path.read_bytes()).hexdigest()
ev_digest=hashlib.sha256(json.dumps(cfg["evaluation"],sort_keys=True).encode()).hexdigest()
# Reuse verified extraction and hashing without writing a successful audit for rejected data.
code=(root/"scripts/extract_P023.py").read_text().split("overlap=[")[0]
ns={};exec(compile(code,"scripts/extract_P023.py","exec"),ns)
records=ns["records"]
eval_pixels={im["pixel_sha256"] for r in records if r["split"]=="evaluation" for im in r["images"]}
bad=[r for r in records if r["split"]=="calibration" and any(im["pixel_sha256"] in eval_pixels for im in r["images"])]
assert len(bad)==1
old=bad[0]
train=json.loads((root/"data/ImageCoDe/github/data/train_data.json").read_text())
used={r["cluster"] for r in cfg["calibration"]}
gold=str(old["gold"])
options=[{"id":k+":"+gold,"directory":k,"cluster":k.split("-shot")[0],
          "kind":"static" if k.startswith("open-images-") else "video",
          "gold":int(gold),"caption":v[gold],"split":"calibration"}
         for k,v in train.items() if gold in v and k.split("-shot")[0] not in used]
options=[r for r in options if r["kind"]==old["kind"]]
options.sort(key=lambda r:hashlib.sha256((cfg["seed"]+":"+r["id"]).encode()).hexdigest())
all_other_pixels={im["pixel_sha256"] for r in records if r["id"]!=old["id"] for im in r["images"]}
with zipfile.ZipFile(root/"data/ImageCoDe/image-sets.zip") as z:
    bydir={}
    for info in z.infolist():
        p=Path(info.filename)
        if not info.is_dir() and p.suffix.lower()==".jpg":
            bydir.setdefault(p.parts[-2],[]).append(info)
    replacement=None
    for r in options:
        infos=bydir.get(r["directory"],[])
        if len(infos)!=10:continue
        pixels=[]
        for info in infos:
            with Image.open(io.BytesIO(z.read(info))) as im:
                im=im.convert("RGB")
                pixels.append(hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest())
        if not set(pixels)&all_other_pixels:
            replacement=r;break
assert replacement is not None
for i,r in enumerate(cfg["calibration"]):
    if r["id"]==old["id"]:cfg["calibration"][i]=replacement
assert len(cfg["calibration"])==40
assert all(sum(r["gold"]==g for r in cfg["calibration"])==4 for g in range(10))
assert hashlib.sha256(json.dumps(cfg["evaluation"],sort_keys=True).encode()).hexdigest()==ev_digest
amendment={
    "reason":"One exact RGB pixel image shared between calibration and evaluation in official train/valid data.",
    "removed_calibration_id":old["id"],
    "replacement_calibration_id":replacement["id"],
    "selection_rule":"First original-seed hash-ranked unused training source cluster with same target index and static/video kind, whose 10 images have no exact RGB pixel overlap with retained data.",
    "original_config_sha256":original_cfg_sha,
    "unchanged_evaluation_sha256":ev_digest,
    "n_evaluation":256,
    "before_any_P023_evaluation_inference":True,
    "source":"scripts/amend_P023_dedup.py"
}
cfg["data_integrity_amendment"]=amendment
cfg_path.write_text(json.dumps(cfg,indent=2))
amendment["amended_config_sha256"]=hashlib.sha256(cfg_path.read_bytes()).hexdigest()
(root/"results/P023_data_amendment.json").write_text(json.dumps(amendment,indent=2))
print(json.dumps(amendment,indent=2))

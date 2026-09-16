"""Extract only selected ImageCoDe sets from the pinned, checksum-verified archive."""
import json,hashlib,zipfile,collections
from pathlib import Path
from PIL import Image
ROOT=Path('/vepfs/vlms/1');assert Path.cwd().resolve()==ROOT
cfg=json.loads(Path('configs/P023_imagecode.json').read_text())
data=ROOT/'data/ImageCoDe'
v=json.loads((data/'verification.json').read_text());assert v['complete']
m=json.loads((data/'source_manifest.json').read_text())
entry=next(x for x in v['files'] if x['path']=='image-sets.zip')
assert entry['sha256']==next(x for x in m['files'] if x['path']=='image-sets.zip')['lfs']['sha256']
wanted={r['directory'] for r in cfg['calibration']+cfg['evaluation']}
dest=data/'selected_P023';dest.mkdir(exist_ok=True)
image_files=collections.defaultdict(list)
with zipfile.ZipFile(data/'image-sets.zip') as z:
 for info in z.infolist():
  p=Path(info.filename)
  if info.is_dir() or len(p.parts)<2 or p.parts[-2] not in wanted or p.suffix.lower() not in {'.jpg','.jpeg','.png'}:continue
  q=(dest/p.parts[-2]/p.name).resolve();assert q.is_relative_to(dest)
  q.parent.mkdir(exist_ok=True)
  if not q.exists():q.write_bytes(z.read(info))
  with Image.open(q) as im:im.verify()
  image_files[p.parts[-2]].append(q)
records=[];hash_splits=collections.defaultdict(set);hash_groups=collections.defaultdict(set)
for r in cfg['calibration']+cfg['evaluation']:
 paths=sorted(image_files[r['directory']],key=lambda p:int(p.stem.replace('img','')))
 # Video filenames encode source frame numbers; the gold is the rank after numeric sorting.
 # Official baselines/clip/contextual.py at 609f076 lines 175-190 uses this mapping.
 assert len(paths)==10 and len(set(paths))==10,r['directory']
 images=[]
 for p in paths:
  bytehash=hashlib.sha256(p.read_bytes()).hexdigest()
  with Image.open(p) as im:
   im=im.convert('RGB');pixelhash=hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest();size=im.size
  images.append({'path':str(p.relative_to(ROOT)),'sha256':bytehash,'pixel_sha256':pixelhash,'size':size})
  hash_splits[pixelhash].add(r['split']);hash_groups[pixelhash].add(r['cluster'])
 records.append({**r,'images':images})
overlap=[h for h,ss in hash_splits.items() if len(ss)>1]
audit={'n_records':len(records),'n_unique_pixel_images':len(hash_splits),'calibration_eval_pixel_overlaps':len(overlap),
 'shared_images_across_groups':sum(len(g)>1 for g in hash_groups.values()),'archive_sha256':entry['sha256'],
 'image_order':'numeric source frame suffix, as in the official baseline; gold is sorted-list index',
 'noncontiguous_filename_sets':sum([Path(im['path']).stem for im in r['images']]!=['img'+str(i) for i in range(10)] for r in records),
 'official_mapping_source':'https://raw.githubusercontent.com/McGill-NLP/imagecode/609f07611aed2599f946c30d730f40a41af1079b/baselines/clip/contextual.py'}
# Cross-group image duplicates are retained in metadata for cluster-aware analysis.
assert not overlap,audit
out=Path('runs/P023_data');out.mkdir(exist_ok=True)
(out/'records.json').write_text(json.dumps(records,indent=2))
(out/'audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit))

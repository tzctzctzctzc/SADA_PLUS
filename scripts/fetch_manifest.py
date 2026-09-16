import argparse,hashlib,json,os,time
from pathlib import Path
from huggingface_hub import snapshot_download
root=Path('/vepfs/vlms/1')
assert Path.cwd().resolve()==root
p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--destination',required=True)
args=p.parse_args();dest=Path(args.destination).resolve();assert dest.is_relative_to(root)
m=json.loads(Path(args.manifest).read_text());dest.mkdir(parents=True,exist_ok=True)
(dest/'source_manifest.json').write_text(json.dumps(m,indent=2)+'\n')
patterns=['*.json','*.safetensors','*.txt','*.model','*.parquet']
print('START',m['repository'],m['revision'],flush=True)
snapshot_download(repo_id=m['repository'],repo_type=m['type'],revision=m['revision'],local_dir=dest,
 allow_patterns=patterns,max_workers=4,token=False)
checked=[]
for row in m['files']:
 q=dest/row['path']
 if not q.exists():continue
 assert q.stat().st_size==row['size'],row['path']
 h=hashlib.sha256()
 with q.open('rb') as f:
  while chunk:=f.read(8*1024*1024):h.update(chunk)
 digest=h.hexdigest()
 if row.get('lfs'):assert digest==row['lfs']['sha256'],row['path']
 checked.append({'path':row['path'],'bytes':q.stat().st_size,'sha256':digest})
(dest/'verification.json').write_text(json.dumps({'complete':True,'revision':m['revision'],'files':checked},indent=2)+'\n')
print('VERIFIED',m['repository'],len(checked),flush=True)

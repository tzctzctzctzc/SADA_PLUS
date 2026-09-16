"""Official SigLIP2 forward as an independent-image retrieval control on P023."""
import json,os,time,hashlib,transformers
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor,AutoModel
ROOT=Path('/vepfs/vlms/1');assert Path.cwd().resolve()==ROOT
assert os.environ['CUDA_VISIBLE_DEVICES']=='GPU-7ce1c8eb-f665-99cf-2541-3c905a3f98f1' and torch.cuda.device_count()==1
cfg=json.loads(Path('configs/P023_imagecode.json').read_text());rows=json.loads(Path('runs/P023_data/records.json').read_text())
path=ROOT/'models'/cfg['retrieval_model']
processor=AutoProcessor.from_pretrained(path,local_files_only=True)
model=AutoModel.from_pretrained(path,local_files_only=True,torch_dtype=torch.float32).to('cuda:0').eval()
out=Path('runs/P023_imagecode_siglip');out.mkdir(exist_ok=True)
meta={'config':cfg,'model_revision':json.loads((path/'source_manifest.json').read_text())['revision'],'dtype':'float32','transformers':transformers.__version__,'torch':torch.__version__,
 'source_hashes':{n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in [__file__,'runs/P023_data/records.json']},'text_max_length':64,'score':'official logits_per_image'}
mp=out/'config.json'
if mp.exists():assert json.loads(mp.read_text())==meta
else:mp.write_text(json.dumps(meta,indent=2))
rp=out/'predictions.jsonl';done={r['id'] for r in map(json.loads,rp.read_text().splitlines())} if rp.exists() else set()
torch.backends.cuda.matmul.allow_tf32=False
with torch.inference_mode(),rp.open('a',buffering=1) as f:
 for row in rows:
  if row['id'] in done:continue
  start=time.monotonic();images=[Image.open(ROOT/x['path']).convert('RGB') for x in row['images']]
  text_len=len(processor.tokenizer(row['caption'],truncation=False)['input_ids'])
  inputs=processor(images=images,text=[row['caption']],padding='max_length',max_length=64,truncation=True,return_tensors='pt').to('cuda:0')
  logits=model(**inputs).logits_per_image[:,0].float().cpu()
  r={'id':row['id'],'split':row['split'],'cluster':row['cluster'],'kind':row['kind'],'gold':row['gold'],'scores':logits.tolist(),
    'prediction':int(logits.argmax()),'text_tokens_untruncated':text_len,'truncated':text_len>64,'seconds':time.monotonic()-start}
  f.write(json.dumps(r)+'\n');done.add(row['id'])
  if len(done)%10==0:print(json.dumps({'progress':len(done),'total':len(rows)}),flush=True)
  del inputs,images
(out/'summary.json').write_text(json.dumps({'complete':len(done)==len(rows),'n':len(done)}))
print('COMPLETE',len(done),flush=True)

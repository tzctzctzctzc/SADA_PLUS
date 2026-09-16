"""Independent image-text retrieval baseline; each image is encoded once."""
import io,json,hashlib,os,re,time
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
import torch
from transformers import AutoModel,AutoProcessor
ROOT=Path('/vepfs/vlms/1')
assert Path.cwd()==ROOT and os.environ['CUDA_VISIBLE_DEVICES']=='GPU-7ce1c8eb-f665-99cf-2541-3c905a3f98f1'
assert torch.cuda.device_count()==1
cfg=json.loads(Path('configs/P020_siglip_retrieval.json').read_text())
reference=Path('configs/P015_lad_bridge.json')
assert hashlib.sha256(reference.read_bytes()).hexdigest()==cfg['data_config_sha256']
dc=json.loads(reference.read_text())
mp=ROOT/'models'/cfg['model']
verification=json.loads((mp/'verification.json').read_text())
assert verification['complete'] and verification['revision']==cfg['model_revision']
processor=AutoProcessor.from_pretrained(mp,local_files_only=True)
model=AutoModel.from_pretrained(mp,local_files_only=True,torch_dtype=torch.float32,attn_implementation='sdpa').to('cuda:0').eval()
table=pa.concat_tables([pq.read_table(p) for p in sorted(Path('data/Mantis-Eval').rglob('*.parquet'))])
docs=[];unique={}
for split in ['calibration','evaluation']:
 for row in dc[split]:
  d=table.slice(row['index'],1).to_pylist()[0];assert d['id']==row['id']
  hashes=[]
  for image,expected in zip(d['images'],row['image_hashes']):
   blob=image['bytes'];digest=hashlib.sha256(blob).hexdigest();assert digest==expected
   hashes.append(digest)
   if digest not in unique:unique[digest]=Image.open(io.BytesIO(blob)).convert('RGB')
  gold_letter=ord(d['answer'].strip('() \n'))-65
  gold=int(re.search(r'image\s*(\d+)',d['options'][gold_letter],re.I).group(1))-1
  docs.append({'base_id':row['id'],'split':split,'caption':row['caption'],'hashes':hashes,'gold_original':gold})
torch.backends.cuda.matmul.allow_tf32=False
torch.cuda.reset_peak_memory_stats()
start=time.monotonic();image_features={};text_features={}
keys=list(unique)
with torch.inference_mode():
 for offset in range(0,len(keys),cfg['image_batch_size']):
  kk=keys[offset:offset+cfg['image_batch_size']]
  batch=processor(images=[unique[k] for k in kk],return_tensors='pt').to('cuda:0')
  features=torch.nn.functional.normalize(model.get_image_features(**batch).float(),dim=-1).cpu()
  image_features.update({k:f for k,f in zip(kk,features)})
  del batch,features
 torch.cuda.synchronize();image_seconds=time.monotonic()-start
 start=time.monotonic()
 captions=[d['caption'] for d in docs]
 tokens=processor.tokenizer(captions,padding='max_length',max_length=64,truncation=True,return_tensors='pt').to('cuda:0')
 tf=torch.nn.functional.normalize(model.get_text_features(**tokens).float(),dim=-1).cpu()
 text_features={d['base_id']:f for d,f in zip(docs,tf)}
 torch.cuda.synchronize();text_seconds=time.monotonic()-start
 scale=float(model.logit_scale.exp());bias=float(model.logit_bias)
 d=docs[0]
 check=processor(text=[d['caption']],images=[unique[h] for h in d['hashes']],padding='max_length',max_length=64,truncation=True,return_tensors='pt').to('cuda:0')
 direct=model(**check).logits_per_image[:,0].float().cpu()
 manual=torch.stack([image_features[h] for h in d['hashes']])@text_features[d['base_id']]*scale+bias
 err=float((direct-manual).abs().max())
 assert err<1e-3 and direct.argmax()==manual.argmax()
out=Path('runs/P020_siglip_retrieval');out.mkdir(exist_ok=True)
meta={'config':cfg,'torch':torch.__version__,'model_config':model.config.to_dict(),
 'processor':processor.image_processor.to_dict(),'n_unique_images':len(unique),'n_captions':len(docs),
 'image_seconds':image_seconds,'text_seconds':text_seconds,'peak_memory_gb':torch.cuda.max_memory_allocated()/1e9,
 'model_forward_vs_feature_reconstruction_max_error':err,
 'truncated_captions':sum(len(processor.tokenizer(d['caption'])['input_ids'])>64 for d in docs),
 'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(out/'config.json').write_text(json.dumps(meta,indent=2,default=str)+'\n')
records=[]
for d in docs:
 z=(torch.stack([image_features[h] for h in d['hashes']])@text_features[d['base_id']]*scale+bias).numpy()
 for shift,perm in enumerate(dc['permutations']):
  scores=z[perm];pred=int(scores.argmax());gold=perm.index(d['gold_original'])
  records.append({'id':d['base_id']+':p'+str(shift),'base_id':d['base_id'],'split':d['split'],
   'image_order':perm,'gold':gold,'scores':scores.tolist(),'prediction':pred,
   'predicted_original_image':perm[pred],'correct':pred==gold})
(out/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
(out/'summary.json').write_text(json.dumps({'complete':True,'n_records':len(records),'n_eval':sum(r['split']=='evaluation' for r in records)},indent=2)+'\n')
print(json.dumps({'complete':True,'n_records':len(records),'eval_correct':sum(r['correct'] for r in records if r['split']=='evaluation'),'n_eval':76,'audit_error':err}),flush=True)

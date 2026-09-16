"""Established pointwise matching and caption-likelihood controls; no method claim."""
import json,os,re,hashlib,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from datasets import load_dataset
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
ROOT=Path('/vepfs/vlms/1');assert Path.cwd().resolve()==ROOT
assert os.environ['CUDA_VISIBLE_DEVICES']=='GPU-7ce1c8eb-f665-99cf-2541-3c905a3f98f1' and torch.cuda.device_count()==1
cfg=json.loads(Path('configs/P024_pointwise.json').read_text())
dc=Path(cfg['data_config']);assert hashlib.sha256(dc.read_bytes()).hexdigest()==cfg['data_config_sha256']
data_cfg=json.loads(dc.read_text())
data=load_dataset('parquet',data_files=sorted(str(p) for p in Path('data/Mantis-Eval').rglob('*.parquet')),split='train')
path=ROOT/'models'/cfg['model']
processor=AutoProcessor.from_pretrained(path,local_files_only=True,min_pixels=cfg['min_pixels'],max_pixels=cfg['max_pixels'])
model=Qwen2_5_VLForConditionalGeneration.from_pretrained(path,local_files_only=True,torch_dtype='auto',device_map={'':0},attn_implementation='sdpa').eval()
out=Path('runs/P024_pointwise');out.mkdir(exist_ok=True)
ids=[processor.tokenizer.encode(t,add_special_tokens=False) for t in ['True','False']]
assert all(len(x)==1 for x in ids);ids=[x[0] for x in ids]
meta={'config':cfg,'torch':torch.__version__,'transformers':transformers.__version__,'candidate_ids':ids,
 'model_revision':json.loads((path/'source_manifest.json').read_text())['revision'],
 'source_hashes':{n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in [__file__,cfg['data_config']]}}
mp=out/'config.json'
if mp.exists():assert json.loads(mp.read_text())==meta
else:mp.write_text(json.dumps(meta,indent=2))
rp=out/'predictions.jsonl';done={r['id'] for r in map(json.loads,rp.read_text().splitlines())} if rp.exists() else set()
def inputs_for(image,prompt):
 content=[{'type':'image'},{'type':'text','text':prompt}]
 text=processor.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True)
 batch=processor(text=[text],images=[image],return_tensors='pt')
 return {k:v.to('cuda:0',dtype=model.dtype if v.is_floating_point() else None) for k,v in batch.items()}
torch.backends.cuda.matmul.allow_tf32=False
with torch.inference_mode(),rp.open('a',buffering=1) as f:
 for split in ['calibration','evaluation']:
  for row in data_cfg[split]:
   d=data[row['index']];assert d['id']==row['id']
   opt=ord(d['answer'].strip('() \n'))-65
   gold=int(re.search(r'image\s*(\d+)',d['options'][opt],re.I).group(1))-1
   for i,im in enumerate(d['images']):
    idx=row['id']+':i'+str(i)
    if idx in done:continue
    start=time.monotonic();image=im.convert('RGB')
    prompt=cfg['matching_prompt'].replace('{caption}',row['caption'])
    inputs=inputs_for(image,prompt)
    output=model(**inputs,use_cache=False)
    matching=output.logits[0,-1,ids].float().cpu().tolist();del output
    match_tokens=inputs['input_ids'].shape[1]
    caption_inputs=inputs_for(image,cfg['caption_prompt'])
    prefix_len=caption_inputs['input_ids'].shape[1]
    caption_ids=processor.tokenizer.encode(row['caption'],add_special_tokens=False)
    assert processor.tokenizer.decode(caption_ids)==row['caption']
    suffix=torch.tensor([caption_ids],device='cuda:0')
    caption_inputs['input_ids']=torch.cat([caption_inputs['input_ids'],suffix],dim=1)
    caption_inputs['attention_mask']=torch.cat([caption_inputs['attention_mask'],torch.ones_like(suffix)],dim=1)
    labels=caption_inputs['input_ids'].clone();labels[:,:prefix_len]=-100
    output=model(**caption_inputs,labels=labels,use_cache=False)
    per_token=-F.cross_entropy(output.logits[:,prefix_len-1:-1,:].float().reshape(-1,model.config.vocab_size),suffix.reshape(-1),reduction='none').cpu()
    loss_error=abs(float(output.loss)+float(per_token.mean()));assert loss_error<1e-5
    r={'id':idx,'base_id':row['id'],'image_index':i,'split':split,'gold_original':gold,
      'true_false_logits':matching,'matching_score':matching[0]-matching[1],
      'caption_logprob_mean':float(per_token.mean()),'caption_logprobs':per_token.tolist(),'caption_token_ids':caption_ids,
      'caption_loss_audit_error':loss_error,'image_grid_thw':inputs['image_grid_thw'].cpu().tolist(),
      'matching_input_tokens':match_tokens,'caption_prefix_tokens':prefix_len,'seconds':time.monotonic()-start}
    f.write(json.dumps(r)+'\n');done.add(idx)
    print(json.dumps({'n':len(done),'id':idx}),flush=True)
    del output,inputs,caption_inputs
summary={'complete':len(done)==96,'n':len(done)}
(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)

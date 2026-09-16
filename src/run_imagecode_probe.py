"""ImageCoDe ten-image retrieval with explicit labels and unprocessed candidate logits."""
import json,os,hashlib,time,argparse
from pathlib import Path
import numpy as np
import torch,transformers
from PIL import Image
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
from last_query_attention import LastQueryAttention
ROOT=Path('/vepfs/vlms/1');assert Path.cwd().resolve()==ROOT
assert os.environ['CUDA_VISIBLE_DEVICES']=='GPU-7ce1c8eb-f665-99cf-2541-3c905a3f98f1' and torch.cuda.device_count()==1
cfg=json.loads(Path('configs/P023_imagecode.json').read_text())
parser=argparse.ArgumentParser();parser.add_argument('--records',default='runs/P023_data/records.json');parser.add_argument('--output',default='runs/P023_imagecode_vlm');args=parser.parse_args()
assert Path(args.records).resolve().is_relative_to(ROOT) and Path(args.output).resolve().is_relative_to(ROOT)
rows=json.loads(Path(args.records).read_text());out=Path(args.output);out.mkdir(exist_ok=True)
model_path=ROOT/'models'/cfg['model']
processor=AutoProcessor.from_pretrained(model_path,local_files_only=True,padding_side='left',min_pixels=cfg['min_pixels'],max_pixels=cfg['max_pixels'])
model=Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path,local_files_only=True,torch_dtype='auto',device_map={'':0},attn_implementation='sdpa').eval()
labels=cfg['labels'];candidates=[processor.tokenizer.encode(x,add_special_tokens=False) for x in labels]
assert all(len(x)==1 for x in candidates);candidates=[x[0] for x in candidates]
layers=list(range(cfg['layer_range'][0],cfg['layer_range'][1]+1))
meta={'config':cfg,'torch':torch.__version__,'transformers':transformers.__version__,'model_dtype':str(model.dtype),'candidate_ids':candidates,
 'model_revision':json.loads((model_path/'source_manifest.json').read_text())['revision'],
 'source_hashes':{n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in [__file__,'src/last_query_attention.py',args.records]}}
mp=out/'config.json'
if mp.exists():assert json.loads(mp.read_text())==meta
else:mp.write_text(json.dumps(meta,indent=2))
rp=out/'predictions.jsonl';done={r['id'] for r in map(json.loads,rp.read_text().splitlines())} if rp.exists() else set()
torch.backends.cuda.matmul.allow_tf32=False
with torch.inference_mode(),rp.open('a',buffering=1) as f:
 for row in rows:
  if row['id'] in done:continue
  start=time.monotonic()
  images=[Image.open(ROOT/x['path']).convert('RGB') for x in row['images']]
  content=[]
  for label in labels:content.extend([{'type':'text','text':f'Image {label}: '},{'type':'image'}])
  prompt=f"Identify the image that best matches the following description. Respond with its letter only.\nDescription: {row['caption']}\nAnswer: "
  content.append({'type':'text','text':prompt})
  text=processor.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True)
  batch=processor(text=[text],images=images,padding=True,return_tensors='pt')
  inputs={k:v.to(device='cuda:0',dtype=model.dtype if v.is_floating_point() else None) for k,v in batch.items()}
  probe=LastQueryAttention(model,inputs['input_ids'],layers=layers,keep_parts=True)
  assert len(probe.ranges)==10
  torch.cuda.reset_peak_memory_stats()
  with probe:
   gen=model.generate(**inputs,max_new_tokens=1,do_sample=False,temperature=1.0,num_beams=1,repetition_penalty=1.0,output_scores=True,output_logits=True,return_dict_in_generate=True,output_attentions=False)
  raw=gen.logits[0][0,candidates].float().cpu();processed=gen.scores[0][0,candidates].float().cpu()
  assert torch.equal(raw,processed)
  token=processor.tokenizer.decode(gen.sequences[0,len(inputs['input_ids'][0]):],skip_special_tokens=True)
  r={'id':row['id'],'split':row['split'],'cluster':row['cluster'],'kind':row['kind'],'gold':row['gold'],
    'scores':raw.tolist(),'prediction':int(raw.argmax()),'generated_first_token':token,
    'layer_attention':[probe.scores[l].tolist() for l in layers],
    'parts':{name:[probe.parts[l][i].tolist() for l in layers] for i,name in enumerate(['start','patch','end'])},
    'input_tokens':inputs['input_ids'].shape[1],'image_tokens':[b-a-2 for a,b in probe.ranges],
    'image_grid_thw':inputs['image_grid_thw'].cpu().tolist(),
    'seconds':time.monotonic()-start,'peak_memory_gb':torch.cuda.max_memory_allocated()/1e9}
  f.write(json.dumps(r)+'\n');done.add(row['id'])
  print(json.dumps({'progress':len(done),'total':len(rows),'id':row['id'],'seconds':r['seconds']}),flush=True)
  del gen,probe,inputs,batch,images
summary={'complete':len(done)==len(rows),'n':len(done)}
(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)

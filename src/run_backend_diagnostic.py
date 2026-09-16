"""Locate P015/P017 backend differences with shared inputs and vision-feature swaps."""
import json,os,hashlib,time
from pathlib import Path
import torch,numpy as np
from PIL import Image
from datasets import load_dataset
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
from last_query_attention import LastQueryAttention
ROOT=Path('/vepfs/vlms/1')
assert Path.cwd()==ROOT and os.environ['CUDA_VISIBLE_DEVICES']=='GPU-7ce1c8eb-f665-99cf-2541-3c905a3f98f1'
assert torch.cuda.device_count()==1
cfg=json.loads(Path('configs/P019_backend_diagnostic.json').read_text())
base=json.loads(Path('configs/P015_lad_bridge.json').read_text())
ds=load_dataset('parquet',data_files={'test':sorted(str(p) for p in Path('data/Mantis-Eval').rglob('*.parquet'))},split='test')
processor=AutoProcessor.from_pretrained(ROOT/'models'/base['model'],local_files_only=True,padding_side='left')
models={b:Qwen2_5_VLForConditionalGeneration.from_pretrained(ROOT/'models'/base['model'],local_files_only=True,torch_dtype='auto',device_map={'':0},attn_implementation=b).eval() for b in ['eager','sdpa']}
refs={}
for b,path in cfg['references'].items():
 assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==cfg['reference_sha256'][b]
 refs[b]={r['id']:r for r in map(json.loads,Path(path).read_text().splitlines())}
meta={r['id']:r for r in base['calibration']+base['evaluation']}
candidates=[processor.tokenizer.convert_tokens_to_ids(str(i)) for i in range(1,5)]
layers=list(range(*[base['layer_range'][0],base['layer_range'][1]+1]))
out=Path('runs/P019_backend_diagnostic');out.mkdir(exist_ok=True)
report={'purpose':cfg['purpose'],'records':[]}
torch.backends.cuda.matmul.allow_tf32=False
with torch.inference_mode():
 for key in cfg['ids']:
  q,shift=key.split(':p');row=meta[q];d=ds[row['index']]
  assert d['id']==q
  perm=base['permutations'][int(shift)]
  images=[d['images'][i].convert('RGB').resize((336,336),Image.Resampling.BICUBIC) for i in perm]
  prompt=(f"Given 4 images indexed from 1 to 4, identify the image that best matches the provided caption."
    f" Respond with the index number only and nothing else.\nCaption: {row['caption']}\nAnswer: ")
  messages=[{'role':'user','content':[*[{'type':'image'} for _ in images],{'type':'text','text':prompt}]}]
  text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
  batch=processor(text=[text],images=images,padding=True,return_tensors='pt',max_patches=7)
  inputs={k:v.to('cuda:0',dtype=models['eager'].dtype if v.is_floating_point() else None) for k,v in batch.items()}
  n=inputs['input_ids'].shape[1]
  assert n==refs['eager'][key]['input_tokens']==refs['sdpa'][key]['input_tokens']
  vision={};rr={}
  for mode,backend,override,full in [('eager','eager',None,True),('sdpa','sdpa',None,False),
      ('sdpa_eager_vision','sdpa','eager',False),('eager_sdpa_vision','eager','sdpa',True),
      ('eager_no_output_attn','eager',None,False)]:
   model=models[backend]
   def vh(module,args,output):
    if override is None:
     if mode in ['eager','sdpa']:vision[mode]=output.detach().clone()
     return
    return vision[override]
   h=model.visual.register_forward_hook(vh)
   start=time.monotonic()
   try:
    with LastQueryAttention(model,inputs['input_ids'],layers=layers,keep_parts=True) as probe:
     gen=model.generate(**inputs,max_new_tokens=1,do_sample=False,temperature=1.0,num_beams=1,
       output_scores=True,return_dict_in_generate=True,output_attentions=full)
   finally:h.remove()
   scores=gen.scores[0][0,candidates].float().cpu().numpy()
   raw=processor.tokenizer.decode(gen.sequences[0,n:],skip_special_tokens=True)
   z={'scores':scores.tolist(),'raw':raw,'seconds':time.monotonic()-start}
   ref=np.array(refs[backend][key]['scores']);delta=scores-ref
   z['own_backend_reference']={'same_raw':raw==refs[backend][key]['raw_first_token'],
    'score_max_error':float(np.abs(delta).max()),'centered_score_max_error':float(np.abs(delta-delta.mean()).max())}
   if 'eager' in rr:
    delta=scores-np.array(rr['eager']['scores'])
    z['delta_from_clean_eager']={'common_offset':float(delta.mean()),'centered_max_error':float(np.abs(delta-delta.mean()).max()),
      'max_pairwise_margin_error':float(delta.max()-delta.min())}
   rr[mode]=z;del gen,probe
  report['records'].append({'id':key,'modes':rr,
    'vision_relative_l2':float(torch.linalg.vector_norm((vision['eager']-vision['sdpa']).float())/torch.linalg.vector_norm(vision['eager'].float()))})
  print(json.dumps(report['records'][-1]),flush=True)
  (out/'partial.json').write_text(json.dumps(report,indent=2))
  del inputs,batch,vision
report['complete']=True
report['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
Path('results/P019_backend_diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')

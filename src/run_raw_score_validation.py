"""Validate inverse repetition penalty against raw generation logits on GPU 3."""
import json,os,time,hashlib,gc
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
from interventions import delimiter_intervention
ROOT=Path('/vepfs/vlms/1')
assert Path.cwd().resolve()==ROOT
assert os.environ['CUDA_VISIBLE_DEVICES']=='GPU-7ce1c8eb-f665-99cf-2541-3c905a3f98f1'
assert torch.cuda.device_count()==1
cfg=json.loads(Path('configs/P022_raw_score_validation.json').read_text())
assert hashlib.sha256(Path(cfg['input']).read_bytes()).hexdigest()==cfg['source_hash']
data=load_dataset('parquet',data_files={'test':sorted(str(p) for p in Path('data/Mantis-Eval').rglob('*.parquet'))},split='test')
reconstructed={(r['tag'],r['mode'],r['id']):r for r in json.loads(Path(cfg['input']).read_text())}
out=Path('runs/P022_raw_score_validation');out.mkdir(exist_ok=True)
records=[]
torch.backends.cuda.matmul.allow_tf32=False
for case in cfg['settings']:
 c=json.loads(Path(case['config']).read_text());path=Path('models')/c['model']
 processor=AutoProcessor.from_pretrained(path,local_files_only=True,padding_side='left',**({'min_pixels':c['min_pixels'],'max_pixels':c['max_pixels']} if case['mode']=='native2048' else {}))
 model=Qwen2_5_VLForConditionalGeneration.from_pretrained(path,local_files_only=True,torch_dtype='auto',device_map={'':0},attn_implementation=case['backend']).eval()
 candidates=[processor.tokenizer.convert_tokens_to_ids(str(i)) for i in range(1,5)]
 docs={r['id']:r for r in c['calibration']+c['evaluation']}
 for idx in case['ids']:
  base,shift=idx.split(':p');row=docs[base];d=data[row['index']];assert d['id']==base
  perm=c['permutations'][int(shift)]
  images=[d['images'][i].convert('RGB') for i in perm]
  if case['mode']!='native2048':images=[im.resize((336,336),Image.Resampling.BICUBIC) for im in images]
  prompt=(f"Given 4 images indexed from 1 to 4, identify the image that best matches the provided caption."
       f" Respond with the index number only and nothing else.\nCaption: {row['caption']}\nAnswer: ")
  messages=[{'role':'user','content':[*[{'type':'image'} for _ in images],{'type':'text','text':prompt}]}]
  text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
  batch=processor(text=[text],images=images,padding=True,return_tensors='pt',max_patches=7)
  inputs={k:v.to(device='cuda:0',dtype=model.dtype if v.is_floating_point() else None) for k,v in batch.items()}
  mode=case['mode'] if case['tag']!='P017' else 'baseline'
  with torch.inference_mode(),delimiter_intervention(model,inputs['input_ids'],mode):
   gen=model.generate(**inputs,max_new_tokens=1,do_sample=False,temperature=1.0,num_beams=1,output_scores=True,output_logits=True,return_dict_in_generate=True,output_attentions=case['backend']=='eager')
  raw=gen.logits[0][0,candidates].float().cpu()
  processed=gen.scores[0][0,candidates].float().cpu()
  ref=reconstructed[(case['tag'],case['mode'],idx)]
  inverse=torch.tensor(ref['raw_reconstructed'],dtype=torch.float32)
  rounded=inverse.to(torch.bfloat16).float()
  z=np.array(ref['processed'])
  r={'tag':case['tag'],'mode':case['mode'],'id':idx,'raw':raw.tolist(),'processed':processed.tolist(),
     'model_dtype':str(model.dtype),'raw_dtype':str(gen.logits[0].dtype),
     'raw_is_bfloat16_representable':bool(torch.equal(raw,raw.to(torch.bfloat16).float())),
     'processed_replay_max_error':float(np.max(np.abs(z-processed.numpy()))),
     'inverse_max_error':float((raw-inverse).abs().max()),
     'inverse_bfloat16_max_error':float((raw-rounded).abs().max()),
     'inverse_argmax_matches':int(raw.argmax())==int(inverse.argmax()),
     'inverse_bfloat16_argmax_matches':int(raw.argmax())==int(rounded.argmax()),
     'true_raw_argmax':int(raw.argmax()),'processed_argmax':int(processed.argmax())}
  records.append(r)
  with (out/'predictions.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
  print(json.dumps(r),flush=True)
  del gen,inputs,batch
 del model,processor
 gc.collect();torch.cuda.empty_cache()
assert len(records)==sum(len(x['ids']) for x in cfg['settings'])
report={'complete':True,'n':len(records),'processed_replay_max_error':max(r['processed_replay_max_error'] for r in records),
 'inverse_max_error':max(r['inverse_max_error'] for r in records),
 'inverse_bfloat16_max_error':max(r['inverse_bfloat16_max_error'] for r in records),
 'unrounded_argmax_mismatches':sum(not r['inverse_argmax_matches'] for r in records),
 'rounded_argmax_mismatches':sum(not r['inverse_bfloat16_argmax_matches'] for r in records),
 'all_raw_bfloat16_representable':all(r['raw_is_bfloat16_representable'] for r in records),
 'selection':'Post-hoc diagnostics, largest changed margin, nearest reconstructed tie, and unchanged control per setting.',
 'records':records}
Path('results/P022_raw_score_validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)

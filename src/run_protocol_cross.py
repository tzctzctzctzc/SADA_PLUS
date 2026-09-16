import argparse,hashlib,json,os,re,time,random
from pathlib import Path
import numpy as np
import torch,transformers
from PIL import Image
from datasets import load_dataset
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from interventions import delimiter_intervention
from last_query_attention import LastQueryAttention
from permutation_data import factorial_samples

ROOT=Path('/vepfs/vlms/1');GPU='GPU-7ce1c8eb-f665-99cf-2541-3c905a3f98f1'
assert Path.cwd().resolve()==ROOT and os.environ.get('CUDA_VISIBLE_DEVICES')==GPU
assert torch.cuda.device_count()==1
p=argparse.ArgumentParser();p.add_argument('--model-name',required=True);p.add_argument('--output',required=True)
args=p.parse_args();cfg=json.loads(Path('configs/P008_protocol_cross.json').read_text())
assert args.model_name in cfg['models']
out=Path(args.output).resolve();assert out.is_relative_to(ROOT);out.mkdir(parents=True,exist_ok=True)
ds=load_dataset('parquet',data_files={'test':[str(p) for p in Path('data/Mantis-Eval').rglob('*.parquet')]},split='test')
factor={'ids':cfg['ids'],'image_permutations':cfg['image_permutations'],'option_permutations':[[0,1,2,3]]}
rows=factorial_samples(ds,factor)
model_path=ROOT/'models'/args.model_name
proc={res:AutoProcessor.from_pretrained(model_path,local_files_only=True,**(
    {'min_pixels':cfg['native_pixels']['min'],'max_pixels':cfg['native_pixels']['max']} if res=='native2048' else {})) for res in cfg['preprocessing']}
model=Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path,local_files_only=True,torch_dtype=torch.bfloat16,
    device_map={'':0},attn_implementation='sdpa').eval()
layers=list(range(21 if '3B' in args.model_name else 16,len(model.model.layers)))
meta={'args':vars(args),'config':cfg,'layers':layers,'gpu_uuid':GPU,'torch':torch.__version__,'transformers':transformers.__version__,
 'model_revision':json.loads((model_path/'source_manifest.json').read_text())['revision'],
 'dataset_revision':json.loads(Path('data/Mantis-Eval/source_manifest.json').read_text())['revision'],
 'generation_config':model.generation_config.to_dict(),'sources':{str(q):hashlib.sha256(q.read_bytes()).hexdigest()
  for q in [Path(__file__),Path('src/last_query_attention.py'),Path('src/interventions.py'),Path('src/permutation_data.py')]}}
cp=out/'config.json'
if cp.exists():assert json.loads(cp.read_text())==meta
cp.write_text(json.dumps(meta,indent=2))
rp=out/'predictions.jsonl'
done={r['id']:r for r in map(json.loads,rp.read_text().splitlines())} if rp.exists() else {}
torch.backends.cuda.matmul.allow_tf32=False
jobs=[(m,res,prompt,d) for m in cfg['modes'] for res in cfg['preprocessing'] for prompt in cfg['prompts'] for d in rows]
with torch.inference_mode(),rp.open('a',buffering=1) as f:
 for mode,res,prompt_kind,d in jobs:
    key=f"{mode}:{res}:{prompt_kind}:{d['id']}"
    if key in done:continue
    start=time.monotonic();processor=proc[res];tok=processor.tokenizer
    caption=d['question'].split("Which image best depicts the caption '",1)[1][:-2]
    if prompt_kind=='mantis_letters':
        prompt=(d['question']+"\nAnswer with the option's letter from the given choices directly."+str(d['options'])).replace('<image>','')
    else:
        prompt=("Given 4 images indexed from 1 to 4, identify the image that best matches the provided caption."
            f" Respond with the index number only and nothing else.\nCaption: {caption}\nAnswer: ")
    if res=='native2048':
        content=[{'type':'image','image':im.convert('RGB'),'min_pixels':cfg['native_pixels']['min'],'max_pixels':cfg['native_pixels']['max']} for im in d['images']]
    else:
        content=[{'type':'image','image':im.convert('RGB').resize((336,336),Image.Resampling.BICUBIC)} for im in d['images']]
    messages=([{'role':'system','content':'You are a helpful assistant.'}] if prompt_kind=='mantis_letters' else [])
    messages.append({'role':'user','content':content+[{'type':'text','text':prompt}]})
    text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    if res=='native2048': images,videos=process_vision_info(messages)
    else:images=[r['image'] for r in content];videos=None
    batch=processor(text=[text],images=images,videos=videos,padding=True,return_tensors='pt')
    inputs={k:v.to(device='cuda:0',dtype=model.dtype if v.is_floating_point() else None) for k,v in batch.items()}
    torch.cuda.reset_peak_memory_stats()
    with delimiter_intervention(model,inputs['input_ids'],mode),LastQueryAttention(model,inputs['input_ids'],layers=layers,keep_parts=True) as probe:
        gen=model.generate(**inputs,do_sample=False,temperature=None,top_p=None,max_new_tokens=cfg['max_new_tokens'],
            num_beams=1,use_cache=True,return_dict_in_generate=True,output_scores=True,
            eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id)
    suffix=gen.sequences[0,inputs['input_ids'].shape[1]:]
    raw=tok.decode(suffix,skip_special_tokens=True,clean_up_tokenization_spaces=False)
    if prompt_kind=='mantis_letters':
        parsed=next((c.upper() for c in raw if c.isalpha()),None)
        pred=ord(parsed)-65 if parsed in ['A','B','C','D'] else -1
    else:
        nums=[int(x) for x in re.findall(r'\d+',raw) if 1<=int(x)<=4]
        pred=nums[-1]-1 if nums else -1
    gold=d['image_order'].index(d['gold_original_image'])
    candidates={}
    symbols='ABCD' if prompt_kind=='mantis_letters' else '1234'
    for prefix in ['','(',' ']:
        t=[tok.encode(prefix+c,add_special_tokens=False) for c in symbols]
        if all(len(x)==1 for x in t):candidates[repr(prefix)]=gen.scores[0][0,[x[0] for x in t]].float().cpu().tolist()
    mass=torch.stack([probe.scores[i] for i in layers]).tolist()
    torch.cuda.synchronize()
    record={'id':key,'base_id':d['base_id'],'model':args.model_name,'mode':mode,'preprocessing':res,'prompt':prompt_kind,
        'image_order':d['image_order'],'gold':gold,'prediction':pred,'correct':pred==gold,'raw':raw,
        'output_tokens':len(suffix),'input_tokens':inputs['input_ids'].shape[1],'image_grid_thw':inputs['image_grid_thw'].cpu().tolist(),
        'candidate_scores':candidates,'attention':mass,'attention_parts':torch.stack([probe.parts[i] for i in layers]).tolist(),'seconds':time.monotonic()-start,
        'peak_memory_gb':torch.cuda.max_memory_allocated()/1e9}
    f.write(json.dumps(record)+'\n');done[key]=record
    print(json.dumps({'progress':f'{len(done)}/{len(jobs)}','id':key,'correct':record['correct']}),flush=True)
    del gen,probe,inputs,batch
summary={'n':len(done),'expected':len(jobs),'complete':len(done)==len(jobs)}
(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)

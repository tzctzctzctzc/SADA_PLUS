"""Official-code LAD calibration transferred to disjoint Mantis groups."""
import argparse, hashlib, json, os, re, time
from pathlib import Path
from contextlib import nullcontext
import numpy as np
import torch, transformers
from datasets import load_dataset
from PIL import Image
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
from interventions import delimiter_intervention
from last_query_attention import LastQueryAttention

ROOT=Path('/vepfs/vlms/1')
GPU='GPU-7ce1c8eb-f665-99cf-2541-3c905a3f98f1'
assert Path.cwd().resolve()==ROOT and os.environ['CUDA_VISIBLE_DEVICES']==GPU
assert torch.cuda.device_count()==1
p=argparse.ArgumentParser()
p.add_argument('--mode',choices=['baseline','scale'],required=True)
p.add_argument('--output',required=True)
p.add_argument('--config',default='configs/P017_lad_resolution.json')
p.add_argument('--resolution',choices=['square336','native2048'],required=True)
args=p.parse_args()
out=Path(args.output).resolve(); assert out.is_relative_to(ROOT)
out.mkdir(parents=True,exist_ok=True)
cfg=json.loads((ROOT/args.config).read_text())
layers=list(range(cfg['layer_range'][0],cfg['layer_range'][1]+1))
model_path=ROOT/'models'/cfg['model']
ds=load_dataset('parquet',data_files={'test':sorted(str(p) for p in (ROOT/'data/Mantis-Eval').rglob('*.parquet'))},split='test')
processor=AutoProcessor.from_pretrained(model_path,local_files_only=True,padding_side='left',**({'min_pixels':cfg['min_pixels'],'max_pixels':cfg['max_pixels']} if args.resolution=='native2048' else {}))
model=Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path,local_files_only=True,
    torch_dtype='auto',device_map={'':0},attn_implementation='sdpa').eval()
candidate_ids=[processor.tokenizer.convert_tokens_to_ids(str(i)) for i in range(1,5)]
assert all(i!=processor.tokenizer.unk_token_id for i in candidate_ids)
meta={'args':vars(args),'config':cfg,'torch':torch.__version__,'transformers':transformers.__version__,
    'model_revision':json.loads((model_path/'source_manifest.json').read_text())['revision'],
    'dataset_revision':json.loads((ROOT/'data/Mantis-Eval/source_manifest.json').read_text())['revision'],
    'dtype':str(model.dtype),'gpu_uuid':GPU,'generation_config':model.generation_config.to_dict(),
    'candidate_ids':candidate_ids,'attention_source':'SDPA with last-query attention sidecar; all square rows compared against P015 full eager attention',
    'source_hashes':{str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in [Path(__file__),Path('src/interventions.py'),Path('src/last_query_attention.py')]}}
mp=out/'config.json'
if mp.exists(): assert json.loads(mp.read_text())==meta
mp.write_text(json.dumps(meta,indent=2))
rp=out/'predictions.jsonl'
done={r['id']:r for r in map(json.loads,rp.read_text().splitlines())} if rp.exists() else {}
jobs=[(split,r,shift,perm) for split in ('calibration','evaluation') for r in cfg[split] for shift,perm in enumerate(cfg['permutations'])]
reference_path=Path('runs/P015_lad_baseline/predictions.jsonl')
assert hashlib.sha256(reference_path.read_bytes()).hexdigest()==cfg['reference_predictions_sha256']
reference={r['id']:r for r in map(json.loads,reference_path.read_text().splitlines())}
assert args.mode=='baseline'
torch.backends.cuda.matmul.allow_tf32=False
with torch.inference_mode(),rp.open('a',buffering=1) as f:
 for split,row,shift,perm in jobs:
    idx=f"{row['id']}:p{shift}"
    if idx in done: continue
    start=time.monotonic(); d=ds[row['index']]
    assert d['id']==row['id']
    gold_option=ord(d['answer'].strip('() \n'))-65
    gold_original=int(re.search(r'image\s*(\d+)',d['options'][gold_option],re.I).group(1))-1
    gold=perm.index(gold_original)
    images=[d['images'][i].convert('RGB') for i in perm]
    if args.resolution=='square336':images=[im.resize((336,336),Image.Resampling.BICUBIC) for im in images]
    prompt=(f"Given 4 images indexed from 1 to 4, identify the image that best matches the provided caption."
        f" Respond with the index number only and nothing else.\nCaption: {row['caption']}\nAnswer: ")
    messages=[{'role':'user','content':[*[{'type':'image'} for _ in images],{'type':'text','text':prompt}]}]
    text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    batch=processor(text=[text],images=images,padding=True,return_tensors='pt',max_patches=7)
    inputs={k:v.to(device='cuda:0',dtype=model.dtype if v.is_floating_point() else None) for k,v in batch.items()}
    ids=inputs['input_ids'][0]
    starts=(ids==model.config.vision_start_token_id).nonzero(as_tuple=True)[0].tolist()
    ends=(ids==model.config.vision_end_token_id).nonzero(as_tuple=True)[0].tolist()
    assert len(starts)==len(ends)==4
    probe=LastQueryAttention(model,inputs['input_ids'],layers=layers,keep_parts=True)
    torch.cuda.reset_peak_memory_stats()
    with delimiter_intervention(model,inputs['input_ids'],args.mode), (probe or nullcontext()):
        gen=model.generate(**inputs,max_new_tokens=1,do_sample=False,temperature=1.0,num_beams=1,
            output_scores=True,return_dict_in_generate=True,output_attentions=False)
    masses=[probe.scores[layer].tolist() for layer in layers]
    parts={name:[probe.parts[layer][i].tolist() for layer in layers] for i,name in enumerate(['start','patch','end'])}
    scores=gen.scores[0][0,candidate_ids].float().cpu().tolist()
    raw=processor.tokenizer.decode(gen.sequences[0,len(ids):],skip_special_tokens=True)
    torch.cuda.synchronize()
    r={'id':idx,'base_id':row['id'],'split':split,'image_order':perm,'gold':gold,
       'gold_original':gold_original,'scores':scores,'layer_attention':masses,'parts':parts,
       'raw_first_token':raw,'candidate_argmax':int(np.argmax(scores)),'input_tokens':len(ids),
       'image_grid_thw':inputs['image_grid_thw'].cpu().tolist(),
       'seconds':time.monotonic()-start,'peak_memory_gb':torch.cuda.max_memory_allocated()/1e9}
    if args.resolution=='square336':
        ref=reference[idx]
        assert r['input_tokens']==ref['input_tokens'] and r['image_grid_thw']==ref['image_grid_thw']
        r['eager_reference_audit']={'same_raw':raw==ref['raw_first_token'],
            'score_max_error':float(np.max(np.abs(np.array(scores)-np.array(ref['scores'])))),
            'attention_mass_max_error':float(np.max(np.abs(np.array(masses)-np.array(ref['layer_attention']))))}
    f.write(json.dumps(r)+'\n');done[idx]=r
    print(json.dumps({'progress':f'{len(done)}/{len(jobs)}','id':idx,'mode':args.mode,'correct':r['candidate_argmax']==gold}),flush=True)
    del gen,probe,inputs,batch
summary={'n':len(done),'expected':len(jobs),'complete':len(done)==len(jobs),'mode':args.mode}
(out/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary),flush=True)

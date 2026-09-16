import ast,hashlib,json,os,re,time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import torch,transformers
from datasets import load_dataset
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from boundary_restore import restore_boundaries,boundary_positions
from permutation_data import factorial_samples
from cache_paths import snapshot,hybrid_cache
ROOT=Path('/vepfs/vlms/1');GPU='GPU-7ce1c8eb-f665-99cf-2541-3c905a3f98f1'
assert Path.cwd().resolve()==ROOT and os.environ.get('CUDA_VISIBLE_DEVICES')==GPU
assert torch.cuda.device_count()==1
cfg=json.loads(Path('configs/P013_cache_paths.json').read_text())
out=Path('runs/P013_cache_paths');out.mkdir(parents=True,exist_ok=True)
ds=load_dataset('parquet',data_files={'test':sorted(str(p) for p in Path('data/Mantis-Eval').rglob('*.parquet'))},split='test')
rows=factorial_samples(ds,cfg['factorial'])
model_path=ROOT/'models'/cfg['model']
processor=AutoProcessor.from_pretrained(model_path,min_pixels=256*28*28,max_pixels=cfg['max_pixels'],local_files_only=True)
model=Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path,torch_dtype=torch.bfloat16,
 device_map={'':0},attn_implementation='sdpa',local_files_only=True).eval()
source=Path('src/run_permutation_probe.py')
nodes=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in {'prompt','inputs_for','parse_answer','score'}]
ns={'re':re,'args':SimpleNamespace(dataset='mantis_permutation',max_pixels=cfg['max_pixels']),
 'processor':processor,'process_vision_info':process_vision_info}
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
inputs_for=ns['inputs_for'];score=ns['score']
meta={'config':cfg,'gpu_uuid':GPU,'torch':torch.__version__,'transformers':transformers.__version__,
 'generation_config':model.generation_config.to_dict(),
 'model_revision':json.loads((model_path/'source_manifest.json').read_text())['revision'],
 'dataset_revision':json.loads(Path('data/Mantis-Eval/source_manifest.json').read_text())['revision'],
 'sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
 Path(__file__),source,Path('src/boundary_restore.py'),Path('src/permutation_data.py'),Path('src/cache_paths.py')]}}
mp=out/'config.json'
if mp.exists():assert json.loads(mp.read_text())==meta
mp.write_text(json.dumps(meta,indent=2))
rp=out/'predictions.jsonl'
done={r['id']:r for r in map(json.loads,rp.read_text().splitlines())} if rp.exists() else {}
old={r['mode']+':'+r['sample_id']:r for r in map(json.loads,Path('runs/P011_geometry_transport/predictions.jsonl').read_text().splitlines())}
gkw={'do_sample':False,'temperature':None,'top_p':None,'num_beams':1,'use_cache':True,
 'return_dict_in_generate':True,'output_scores':True,
 'eos_token_id':processor.tokenizer.eos_token_id,'pad_token_id':processor.tokenizer.pad_token_id}
donors={}
for q,info in cfg['source_states'].items():
 p=Path(info['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==info['sha256']
 if str(p) not in donors:donors[str(p)]=torch.load(p,map_location='cuda:0',weights_only=True)['states']
torch.backends.cuda.matmul.allow_tf32=False
def scores_for(values):
 z={}
 for prefix in ('','(',' '):
  ids=[processor.tokenizer.encode(prefix+c,add_special_tokens=False) for c in 'ABCD']
  if all(len(x)==1 for x in ids):z[repr(prefix)]=values[[x[0] for x in ids]].tolist()
 return z
prefix_key=None;packs={};rope_delta=None
with torch.inference_mode(),rp.open('a',buffering=1) as f:
 for d in rows:
  if all(c['id']+':'+d['id'] in done for c in cfg['conditions']):continue
  inputs=inputs_for(d)
  starts=boundary_positions(model,inputs['input_ids'])[0]
  end=int(boundary_positions(model,inputs['input_ids'])[1,-1])+1
  key=d['id'].split(':o')[0]
  if key!=prefix_key:
   packs.clear()
   pre={k:(v[:,:end] if k in ['input_ids','attention_mask'] else v) for k,v in inputs.items()}
   states=donors[cfg['source_states'][d['base_id']]['path']]
   for shift in [None]+cfg['shifts']:
    ctx=nullcontext() if shift is None else restore_boundaries(model,pre['input_ids'],states,'start',shift)
    with ctx:g=model.generate(**pre,max_new_tokens=1,**gkw)
    cache=snapshot(g.past_key_values)
    assert all(k.shape[-2]==end for k,v in cache)
    packs[shift]=cache
    if shift is None:rope_delta=model.rope_deltas.detach().clone()
    del cache,g
   prefix_key=key
  for cond in cfg['conditions']:
   key=cond['id']+':'+d['id']
   if key in done:continue
   part=cond['part'];shift=cond['shift']
   cache=hybrid_cache(packs[None],packs[shift] if shift is not None else packs[None],starts,part)
   model.rope_deltas=rope_delta.clone()
   begin=time.monotonic();torch.cuda.reset_peak_memory_stats()
   g=model.generate(**inputs,past_key_values=cache,max_new_tokens=cfg['max_new_tokens'],**gkw)
   suffix=g.sequences[:,inputs['input_ids'].shape[1]:]
   raw=processor.batch_decode(suffix,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
   pred,correct=score(d,raw);torch.cuda.synchronize()
   scores=scores_for(g.scores[0][0].float().cpu())
   ref=old[('baseline' if part=='baseline' else 'square448:'+str(shift))+':'+d['id']] if part in ['baseline','full'] else None
   err=max(abs(a-b) for fmt,values in scores.items() for a,b in zip(values,ref['candidate_scores'][fmt])) if ref else None
   rec={'id':key,'sample_id':d['id'],'base_id':d['base_id'],'mode':cond['id'],'part':part,'shift':shift,
    'image_order':d['image_order'],'option_order':d['option_order'],'gold_original_image':d['gold_original_image'],
    'answer':d['answer'],'raw_prediction':raw,'parsed_prediction':pred,'correct':bool(correct),
    'candidate_scores':scores,'reference_P011_raw_match':raw==ref['raw_prediction'] if ref else None,
    'reference_P011_candidate_max_error':err,'prefix_tokens':end,'input_tokens':inputs['input_ids'].shape[1],
    'output_tokens':suffix.shape[1],'query_seconds':time.monotonic()-begin,'peak_memory_gb':torch.cuda.max_memory_allocated()/1e9}
   if isinstance(pred,str) and pred in ['A','B','C','D']:
    rec['predicted_image_position']=d['option_order'][ord(pred)-65]
    rec['predicted_original_image']=d['image_order'][rec['predicted_image_position']]
   f.write(json.dumps(rec)+'\n');done[key]=rec
   print(json.dumps({'progress':f'{len(done)}/{cfg["n_records"]}','id':key,'correct':bool(correct),
    'reference_match':rec['reference_P011_raw_match'],'reference_score_error':err}),flush=True)
   del g,suffix,cache
  del inputs
summary={'n':len(done),'expected':cfg['n_records'],'complete':len(done)==cfg['n_records']}
(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)

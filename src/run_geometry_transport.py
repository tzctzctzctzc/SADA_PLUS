import ast,hashlib,json,os,re,time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import torch,transformers
from datasets import load_dataset
from PIL import Image
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from boundary_restore import CaptureBoundaries,restore_boundaries,boundary_positions
from permutation_data import factorial_samples

ROOT=Path('/vepfs/vlms/1');GPU='GPU-7ce1c8eb-f665-99cf-2541-3c905a3f98f1'
assert Path.cwd().resolve()==ROOT and os.environ.get('CUDA_VISIBLE_DEVICES')==GPU
assert torch.cuda.device_count()==1
cfg=json.loads(Path('configs/P011_geometry_transport.json').read_text())
out=Path('runs/P011_geometry_transport');out.mkdir(parents=True,exist_ok=True)
cache_dir=out/'caches';cache_dir.mkdir(exist_ok=True)
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
 Path(__file__),source,Path('src/boundary_restore.py'),Path('src/permutation_data.py')]}}
mp=out/'config.json'
if mp.exists():assert json.loads(mp.read_text())==meta
mp.write_text(json.dumps(meta,indent=2))
rp=out/'predictions.jsonl'
done={r['id']:r for r in map(json.loads,rp.read_text().splitlines())} if rp.exists() else {}
gkw={'do_sample':False,'temperature':None,'top_p':None,'num_beams':1,'use_cache':True,
 'return_dict_in_generate':True,'output_scores':True,
 'eos_token_id':processor.tokenizer.eos_token_id,'pad_token_id':processor.tokenizer.pad_token_id}
caches={};torch.backends.cuda.matmul.allow_tf32=False
def labeled_inputs_for(d):
 content=[{"type":"image","image":im.convert("RGB"),"min_pixels":256*28*28,"max_pixels":cfg["max_pixels"]} for im in d["images"]]
 messages=[{"role":"system","content":"You are a helpful assistant."},
 {"role":"user","content":content+[{"type":"text","text":ns["prompt"](d)}]}]
 text=processor.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,add_vision_id=True)
 assert "Picture 1:" in text and "Picture 4:" in text
 images,videos=process_vision_info(messages)
 return processor(text=[text],images=images,videos=videos,padding=True,return_tensors="pt").to("cuda:0")

def boundary_coordinates(inputs):
 pos=boundary_positions(model,inputs['input_ids'])
 rope,_=model.get_rope_index(inputs['input_ids'],inputs['image_grid_thw'],attention_mask=inputs['attention_mask'])
 return {'sequence':pos.cpu().tolist(),'rope':rope[:,0,pos].cpu().tolist(),
 'grid':inputs['image_grid_thw'].cpu().tolist()}
def donor_pack(d,inputs,geometry):
 donor_meta=cfg['donors'][d['base_id']]
 donor=ds[donor_meta['index']]
 assert donor['id']==donor_meta['id']
 sizes=[im.size for im in d['images']] if geometry=='matched' else [tuple(cfg['square_size'])]*4
 desc={'id':donor_meta['id'],'sizes':sizes}
 cache_key=hashlib.sha256(json.dumps(desc,sort_keys=True).encode()).hexdigest()
 if cache_key in caches:return caches[cache_key]
 path=cache_dir/(cache_key+'.pt')
 donor_doc=d|{'images':[im.convert('RGB').resize(size,Image.Resampling.BICUBIC) for im,size in zip(donor['images'],sizes)]}
 donor_inputs=inputs_for(donor_doc)
 end=int(boundary_positions(model,donor_inputs['input_ids'])[1,-1])+1
 prefix_hash=hashlib.sha256(donor_inputs['input_ids'][0,:end].cpu().numpy().tobytes()).hexdigest()
 if geometry=='matched':
  assert torch.equal(inputs['input_ids'],donor_inputs['input_ids'])
  assert torch.equal(inputs['image_grid_thw'],donor_inputs['image_grid_thw'])
 if path.exists():
  pack=torch.load(path,map_location='cuda:0',weights_only=True)
  assert pack['prefix_hash']==prefix_hash
 else:
  with CaptureBoundaries(model,donor_inputs['input_ids']) as cap:
   g=model.generate(**donor_inputs,max_new_tokens=1,**gkw)
  pack={'states':cap.states,'prefix_hash':prefix_hash,'coordinates':boundary_coordinates(donor_inputs)}
  cpu_pack=pack|{'states':{i:t.cpu() for i,t in pack['states'].items()}}
  torch.save(cpu_pack,path)
  del cap,g
 caches[cache_key]=pack
 del donor_inputs
 return pack
def letter_scores(values):
 result={}
 for prefix in ('','(',' '):
  ids=[processor.tokenizer.encode(prefix+c,add_special_tokens=False) for c in 'ABCD']
  if all(len(x)==1 for x in ids):result[repr(prefix)]=values[[x[0] for x in ids]].tolist()
 return result
with torch.inference_mode(),rp.open('a',buffering=1) as f:
 for d in rows:
  if all(m+':'+d['id'] in done for m in cfg['modes']):continue
  inputs=inputs_for(d);coords=boundary_coordinates(inputs)
  packs={g:donor_pack(d,inputs,g) for g in cfg['geometries']}
  for mode in cfg['modes']:
   key=mode+':'+d['id']
   if key in done:continue
   geometry,shift=(None,None) if mode.startswith('baseline') else mode.split(':')
   active_inputs=labeled_inputs_for(d) if mode=='baseline_labeled' else inputs
   ctx=nullcontext() if geometry is None else restore_boundaries(model,inputs['input_ids'],packs[geometry]['states'],'start',int(shift))
   start=time.monotonic();torch.cuda.reset_peak_memory_stats()
   with ctx:g=model.generate(**active_inputs,max_new_tokens=cfg['max_new_tokens'],**gkw)
   suffix=g.sequences[:,active_inputs['input_ids'].shape[1]:]
   raw=processor.batch_decode(suffix,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
   pred,correct=score(d,raw);torch.cuda.synchronize()
   record={'id':key,'sample_id':d['id'],'base_id':d['base_id'],'mode':mode,
    'geometry':geometry,'shift':int(shift) if shift is not None else None,
    'image_order':d['image_order'],'option_order':d['option_order'],'gold_original_image':d['gold_original_image'],
    'answer':d['answer'],'raw_prediction':raw,'parsed_prediction':pred,'correct':bool(correct),
    'candidate_scores':letter_scores(g.scores[0][0].float().cpu()),
    'target_coordinates':boundary_coordinates(active_inputs) if mode=='baseline_labeled' else coords,'donor_coordinates':packs[geometry]['coordinates'] if geometry else None,
    'donor_id':cfg['donors'][d['base_id']]['id'],
    'input_tokens':active_inputs['input_ids'].shape[1],'output_tokens':suffix.shape[1],
    'seconds':time.monotonic()-start,'peak_memory_gb':torch.cuda.max_memory_allocated()/1e9}
   if isinstance(pred,str) and pred in ['A','B','C','D']:
    record['predicted_image_position']=d['option_order'][ord(pred)-65]
    record['predicted_original_image']=d['image_order'][record['predicted_image_position']]
   f.write(json.dumps(record)+'\n');done[key]=record
   print(json.dumps({'progress':f'{len(done)}/{cfg["n_records"]}','id':key,'correct':bool(correct)}),flush=True)
   del g,suffix,active_inputs
  del inputs
summary={'n':len(done),'expected':cfg['n_records'],'complete':len(done)==cfg['n_records']}
(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)

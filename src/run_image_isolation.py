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
from image_isolation import isolate_images

ROOT=Path('/vepfs/vlms/1');GPU='GPU-7ce1c8eb-f665-99cf-2541-3c905a3f98f1'
assert Path.cwd().resolve()==ROOT and os.environ.get('CUDA_VISIBLE_DEVICES')==GPU
assert torch.cuda.device_count()==1
cfg=json.loads(Path('configs/P014_image_isolation.json').read_text())
out=Path('runs/P014_image_isolation');out.mkdir(parents=True,exist_ok=True)
cache_dir=Path('runs/P011_geometry_transport/caches')
for path,digest in cfg['donor_cache_sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
reference_path=Path('runs/P011_geometry_transport/predictions.jsonl')
assert hashlib.sha256(reference_path.read_bytes()).hexdigest()==cfg['reference_predictions_sha256']
reference_rows=list(map(json.loads,reference_path.read_text().splitlines()))
reference={m:{r['sample_id']:r for r in reference_rows if r['mode']==m} for m in ['baseline','baseline_labeled']}
ref_meta=json.loads(Path('runs/P011_geometry_transport/config.json').read_text())
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
 Path(__file__),source,Path('src/boundary_restore.py'),Path('src/permutation_data.py'),Path('src/image_isolation.py')]}}
for path in ['src/run_permutation_probe.py','src/permutation_data.py','src/boundary_restore.py']:
 assert meta['sources'][path]==ref_meta['sources'][path]
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
 assert path.exists() and str(path) in cfg['donor_cache_sha256']
 pack=torch.load(path,map_location='cuda:0',weights_only=True)
 assert pack['prefix_hash']==prefix_hash
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
  pack=donor_pack(d,inputs,'square448')
  for mode in cfg['modes']:
   key=mode+':'+d['id']
   if key in done:continue
   labeled=mode=='isolated_labeled'
   active_inputs=labeled_inputs_for(d) if labeled else inputs
   active_coords=boundary_coordinates(active_inputs) if labeled else coords
   ref=reference['baseline_labeled' if labeled else 'baseline'][d['id']]
   assert active_inputs['input_ids'].shape[1]==ref['input_tokens']
   assert active_coords==ref['target_coordinates']
   ctx=restore_boundaries(model,active_inputs['input_ids'],pack['states'],'start',0) if mode=='template' else nullcontext()
   start=time.monotonic();torch.cuda.reset_peak_memory_stats()
   with isolate_images(model,active_inputs['input_ids'],skeleton=mode=='skeleton',block_cross_image=mode!='full_mask'),ctx:
    g=model.generate(**active_inputs,max_new_tokens=cfg['max_new_tokens'],**gkw)
   suffix=g.sequences[:,active_inputs['input_ids'].shape[1]:]
   raw=processor.batch_decode(suffix,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
   pred,correct=score(d,raw);torch.cuda.synchronize()
   record={'id':key,'sample_id':d['id'],'base_id':d['base_id'],'mode':mode,
    'image_order':d['image_order'],'option_order':d['option_order'],'gold_original_image':d['gold_original_image'],
    'answer':d['answer'],'raw_prediction':raw,'parsed_prediction':pred,'correct':bool(correct),
    'candidate_scores':letter_scores(g.scores[0][0].float().cpu()),'target_coordinates':active_coords,
    'donor_id':cfg['donors'][d['base_id']]['id'] if mode=='template' else None,
    'input_ids_sha256':hashlib.sha256(active_inputs['input_ids'].cpu().numpy().tobytes()).hexdigest(),
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

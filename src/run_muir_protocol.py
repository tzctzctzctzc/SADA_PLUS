import ast,hashlib,json,os,random,re,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch,transformers
from datasets import load_dataset
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from boundary_interventions import boundary_intervention
from muir_prompts import content_for

ROOT=Path('/vepfs/vlms/1');GPU='GPU-7ce1c8eb-f665-99cf-2541-3c905a3f98f1'
assert Path.cwd().resolve()==ROOT and os.environ.get('CUDA_VISIBLE_DEVICES')==GPU
assert torch.cuda.device_count()==1
cfg=json.loads(Path('configs/P010_muir_protocol.json').read_text())
out=Path('runs/P010_muir_protocol');out.mkdir(parents=True,exist_ok=True)
ds=load_dataset('parquet',data_files={'test':sorted(str(p) for p in Path('data/MUIRBENCH').rglob('*.parquet'))},split='test')
allow=set(json.loads(Path('configs/muir_split_v1.json').read_text())['dev_ids'])
ds=ds.select([i for i,key in enumerate(ds['idx']) if key in allow])
for r in cfg['rows']:assert ds['idx'][r['dev_index']]==r['id']
model_path=ROOT/'models'/cfg['model']
processor=AutoProcessor.from_pretrained(model_path,min_pixels=256*28*28,max_pixels=cfg['max_pixels'],local_files_only=True)
model=Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path,torch_dtype=torch.bfloat16,
 device_map={'':0},attn_implementation='sdpa',local_files_only=True).eval()
source=Path('src/run_multi_image.py')
nodes=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in {'score','parse_answer'}]
ns={'re':re,'args':SimpleNamespace(dataset='muirbench')}
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
score=ns['score']
meta={'config':cfg,'gpu_uuid':GPU,'torch':torch.__version__,'transformers':transformers.__version__,
 'generation_config':model.generation_config.to_dict(),
 'model_revision':json.loads((model_path/'source_manifest.json').read_text())['revision'],
 'dataset_revision':json.loads(Path('data/MUIRBENCH/source_manifest.json').read_text())['revision'],
 'sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
 Path(__file__),source,Path('src/muir_prompts.py'),Path('src/boundary_interventions.py'),Path('src/interventions.py')]}}
mp=out/'config.json'
if mp.exists():assert json.loads(mp.read_text())==meta
mp.write_text(json.dumps(meta,indent=2))
rp=out/'predictions.jsonl'
done={r['id']:r for r in map(json.loads,rp.read_text().splitlines())} if rp.exists() else {}
torch.backends.cuda.matmul.allow_tf32=False
with torch.inference_mode(),rp.open('a',buffering=1) as f:
 for row in cfg['rows']:
  if all(p+':'+m+':'+row['id'] in done for p in cfg['protocols'] for m in cfg['modes']):continue
  d=ds[row['dev_index']]
  d['image_list']=[im.convert('RGB') for im in d['image_list']]
  for protocol in cfg['protocols']:
   content=content_for(d,protocol,cfg['max_pixels'])
   messages=[{'role':'system','content':'You are a helpful assistant.'},{'role':'user','content':content}]
   text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
   images,videos=process_vision_info(messages)
   inputs=processor(text=[text],images=images,videos=videos,padding=True,return_tensors='pt').to('cuda:0')
   for mode in cfg['modes']:
    key=protocol+':'+mode+':'+d['idx']
    if key in done:continue
    seed=cfg['seed']+row['dev_index']
    torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
    start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    with boundary_intervention(model,inputs['input_ids'],mode):
     g=model.generate(**inputs,do_sample=True,temperature=0.2,top_p=None,
      max_new_tokens=cfg['max_new_tokens'],num_beams=1,use_cache=True,
      eos_token_id=processor.tokenizer.eos_token_id,pad_token_id=processor.tokenizer.pad_token_id)
    suffix=g[:,inputs['input_ids'].shape[1]:]
    raw=processor.batch_decode(suffix,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
    pred,correct=score(d,raw);torch.cuda.synchronize()
    rec={'id':key,'sample_id':d['idx'],'protocol':protocol,'mode':mode,'dev_index':row['dev_index'],
     'seed':seed,'task':d['task'],'counterpart_idx':d['counterpart_idx'],'answer':d['answer'],
     'raw_prediction':raw,'parsed_prediction':pred,'correct':bool(correct),
     'input_tokens':inputs['input_ids'].shape[1],'output_tokens':suffix.shape[1],
     'seconds':time.monotonic()-start,'peak_memory_gb':torch.cuda.max_memory_allocated()/1e9}
    f.write(json.dumps(rec)+'\n');done[key]=rec
    print(json.dumps({'progress':f'{len(done)}/{len(cfg["rows"])*4}','id':key,'correct':bool(correct)}),flush=True)
    del g,suffix
   del inputs
summary={'n':len(done),'expected':len(cfg['rows'])*4,'complete':len(done)==len(cfg['rows'])*4}
(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)

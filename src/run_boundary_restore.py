import argparse,ast,hashlib,json,os,re,time
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
import torch,transformers
from datasets import load_dataset
from PIL import Image
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from interventions import delimiter_intervention
from permutation_data import factorial_samples
from boundary_restore import CaptureBoundaries,restore_boundaries

ROOT=Path('/vepfs/vlms/1');GPU='GPU-7ce1c8eb-f665-99cf-2541-3c905a3f98f1'
assert Path.cwd().resolve()==ROOT and os.environ.get('CUDA_VISIBLE_DEVICES')==GPU
assert torch.cuda.device_count()==1
cfg=json.loads(Path('configs/P009_boundary_restore.json').read_text())
out=Path('runs/P009_boundary_restore');out.mkdir(parents=True,exist_ok=True)
cache_dir=out/'caches';cache_dir.mkdir(exist_ok=True)
ds=load_dataset('parquet',data_files={'test':[str(p) for p in Path('data/Mantis-Eval').rglob('*.parquet')]},split='test')
rows=factorial_samples(ds,cfg['factorial'])
model_path=ROOT/'models'/cfg['model']
processor=AutoProcessor.from_pretrained(model_path,min_pixels=256*28*28,max_pixels=cfg['max_pixels'],local_files_only=True)
model=Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path,torch_dtype=torch.bfloat16,device_map={'':0},
 attn_implementation='sdpa',local_files_only=True).eval()
# Frozen prompt, preprocessing and scorer from the already audited P006 runner.
source=Path('src/run_permutation_probe.py')
names={'prompt','inputs_for','parse_answer','score'}
nodes=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
namespace={'re':re,'args':SimpleNamespace(dataset='mantis_permutation',max_pixels=cfg['max_pixels']),
 'processor':processor,'process_vision_info':process_vision_info}
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),namespace)
inputs_for=namespace['inputs_for'];score=namespace['score']
meta={'config':cfg,'gpu_uuid':GPU,'model_revision':json.loads((model_path/'source_manifest.json').read_text())['revision'],
 'dataset_revision':json.loads(Path('data/Mantis-Eval/source_manifest.json').read_text())['revision'],
 'torch':torch.__version__,'transformers':transformers.__version__,'generation_config':model.generation_config.to_dict(),
 'sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),source,Path('src/interventions.py'),Path('src/boundary_restore.py'),Path('src/permutation_data.py')]}}
mp=out/'config.json'
if mp.exists():assert json.loads(mp.read_text())==meta,'Resume requires identical code/config'
mp.write_text(json.dumps(meta,indent=2))
rp=out/'predictions.jsonl'
done={r['id']:r for r in map(json.loads,rp.read_text().splitlines())} if rp.exists() else {}
gkw={'do_sample':False,'temperature':None,'top_p':None,'num_beams':1,'use_cache':True,'return_dict_in_generate':True,
 'output_scores':True,'eos_token_id':processor.tokenizer.eos_token_id,'pad_token_id':processor.tokenizer.pad_token_id}
caches={};torch.backends.cuda.matmul.allow_tf32=False
def first_score_formats(values):
 ans={}
 for prefix in ('','(',' '):
  ids=[processor.tokenizer.encode(prefix+c,add_special_tokens=False) for c in 'ABCD']
  if all(len(x)==1 for x in ids):ans[repr(prefix)]=values[[x[0] for x in ids]].tolist()
 return ans
with torch.inference_mode(),rp.open('a',buffering=1) as f:
 for d in rows:
    if all(cond['id']+':'+d['id'] in done for cond in cfg['conditions']):continue
    inputs=inputs_for(d);prefix_key=d['id'].split(':o')[0]
    end=(inputs['input_ids'][0]==model.config.vision_end_token_id).nonzero(as_tuple=True)[0][-1].item()+1
    prefix_hash=hashlib.sha256(inputs['input_ids'][0,:end].cpu().numpy().tobytes()).hexdigest()
    if prefix_key not in caches:
        cache_path=cache_dir/(prefix_key.replace(':','_')+'.pt')
        if cache_path.exists():
            pack=torch.load(cache_path,map_location='cuda:0',weights_only=True)
        else:
            pack={'prefix_hash':prefix_hash,'grid':inputs['image_grid_thw'].cpu().tolist()}
            with CaptureBoundaries(model,inputs['input_ids']) as cap:
                g=model.generate(**inputs,max_new_tokens=1,**gkw)
            pack['clean']=cap.states;del cap,g
            donor_meta=cfg['donors'][d['base_id']];donor=ds[donor_meta['index']]
            assert donor['id']==donor_meta['id']
            donor_doc=d | {'images':[im.convert('RGB').resize(target.size,Image.Resampling.BICUBIC) for im,target in zip(donor['images'],d['images'])]}
            donor_inputs=inputs_for(donor_doc)
            assert torch.equal(inputs['input_ids'],donor_inputs['input_ids'])
            assert torch.equal(inputs['image_grid_thw'],donor_inputs['image_grid_thw'])
            with CaptureBoundaries(model,donor_inputs['input_ids']) as cap:
                g=model.generate(**donor_inputs,max_new_tokens=1,**gkw)
            pack['donor']=cap.states;del cap,g,donor_inputs
            assert all(len(pack[src])==len(model.model.layers) for src in ['clean','donor'])
            cpu_pack={k:({i:t.cpu() for i,t in v.items()} if k in ['clean','donor'] else v) for k,v in pack.items()}
            torch.save(cpu_pack,cache_path)
        assert pack['prefix_hash']==prefix_hash and pack['grid']==inputs['image_grid_thw'].cpu().tolist()
        caches[prefix_key]=pack
    pack=caches[prefix_key]
    baseline_scores=None;baseline_raw=None
    # Baseline rerun is needed only after an interrupted partial row for the oracle comparison.
    for cond in cfg['conditions']:
        key=cond['id']+':'+d['id']
        if key in done and cond['id']!='baseline':continue
        start=time.monotonic();torch.cuda.reset_peak_memory_stats()
        restoration=restore_boundaries(model,inputs['input_ids'],pack[cond['source']],cond['kind'],cond['shift']) if cond['source'] else nullcontext()
        with delimiter_intervention(model,inputs['input_ids'],'scale' if cond['scale'] else 'baseline'),restoration:
            gen=model.generate(**inputs,max_new_tokens=cfg['max_new_tokens'],**gkw)
        scores=gen.scores[0][0].float().cpu()
        suffix=gen.sequences[:,inputs['input_ids'].shape[1]:]
        raw=processor.batch_decode(suffix,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
        pred,correct=score(d,raw)
        if cond['id']=='baseline':baseline_scores=scores.clone();baseline_raw=raw
        oracle_error=float((scores-baseline_scores).abs().max()) if cond['id']=='clean_both' else None
        oracle_match=raw==baseline_raw if cond['id']=='clean_both' else None
        torch.cuda.synchronize()
        record={'id':key,'sample_id':d['id'],'base_id':d['base_id'],'mode':cond['id'],
         'image_order':d['image_order'],'option_order':d['option_order'],'gold_original_image':d['gold_original_image'],
         'answer':d['answer'],'raw_prediction':raw,'parsed_prediction':pred,'correct':bool(correct),
         'candidate_scores':first_score_formats(scores),'donor_id':cfg['donors'][d['base_id']]['id'],
         'oracle_score_max_error':oracle_error,'oracle_generation_match':oracle_match,
         'input_tokens':inputs['input_ids'].shape[1],'output_tokens':suffix.shape[1],
         'seconds':time.monotonic()-start,'peak_memory_gb':torch.cuda.max_memory_allocated()/1e9}
        if isinstance(pred,str) and pred in ['A','B','C','D']:
            record['predicted_image_position']=d['option_order'][ord(pred)-65]
            record['predicted_original_image']=d['image_order'][record['predicted_image_position']]
        if key not in done:
            f.write(json.dumps(record)+'\n');done[key]=record
            print(json.dumps({'progress':f'{len(done)}/{len(rows)*len(cfg["conditions"])}','id':key,'correct':correct,
                'oracle_error':oracle_error}),flush=True)
        del gen,scores,suffix
        if cond['id']=='clean_both':
            assert oracle_error<1e-6 and oracle_match,'Full clean-boundary restoration must recover baseline'
    del inputs
summary={'n':len(done),'expected':len(rows)*len(cfg['conditions']),'complete':len(done)==len(rows)*len(cfg['conditions']),
 'oracle_all_match':all(r['oracle_generation_match'] for r in done.values() if r['mode']=='clean_both')}
(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)

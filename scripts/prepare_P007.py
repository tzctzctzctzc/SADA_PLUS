from pathlib import Path
import json, hashlib, re
from datasets import load_dataset, Sequence, Image
root=Path('/vepfs/vlms/1')
ds=load_dataset('parquet',data_files={'test':[str(p) for p in (root/'data/Mantis-Eval').rglob('*.parquet')]},split='test')
raw=ds.cast_column('images',Sequence(Image(decode=False)))
rows=[]
for i,(idx,q) in enumerate(zip(ds['id'],ds['question'])):
    if "Which image best depicts the caption '" not in q: continue
    d=raw[i]
    if len(d['images'])!=4: continue
    caption=q.split("Which image best depicts the caption '",1)[1]
    assert caption.endswith("'?")
    hashes=[hashlib.sha256(im['bytes']).hexdigest() for im in d['images']]
    rows.append({'id':idx,'index':i,'caption':caption[:-2],'image_hashes':hashes})
eval_ids=json.loads((root/'configs/P006_permutation_factorial.json').read_text())['ids']
protected={h for r in rows if r['id'] in eval_ids for h in r['image_hashes']}
cal=[]; used=set()
for r in sorted(rows,key=lambda r:hashlib.sha256(('P007-v1:'+r['id']).encode()).hexdigest()):
    if set(r['image_hashes']) & (protected|used): continue
    cal.append(r); used.update(r['image_hashes'])
    if len(cal)==5: break
assert len(cal)==5
evaluation=[r for r in rows if not set(r['image_hashes']) & used]
assert set(eval_ids).issubset({r['id'] for r in evaluation})
cfg={'id':'P007','purpose':'Exploratory transfer of official LAD/AGD to Mantis caption retrieval; not a reproduction of paper dataset scores or a new method',
 'model':'Qwen2.5-VL-3B-Instruct','modes':['baseline','scale'],'image_size':[336,336],
 'layer_range':[21,35],'topk':2,'sharpen_power':5,'calibration':cal,'evaluation':evaluation,
 'permutations':[[((i+s)%4) for i in range(4)] for s in range(4)],
 'calibration_selection':'Five hash-ranked image-disjoint groups outside P006; calibration and evaluation share no exact encoded image SHA256',
 'official_commit':'c704b9d93eb207652c0b60a5e2e83bcdcae01508',
 'generation':'greedy, first processed generation score including model default repetition penalty',
 'attentions':'last-query rows, vision_start+patches+vision_end; per-part attribution also saved',
 'comparisons':['candidate-logit argmax','raw attention','prior-corrected attention','official AGD code','patch-only attention diagnostic','cross-mode calibration diagnostic'],
 'statistics':'Base questions and shared-image components are units; permutations are dependent. Small post-hoc Mantis subset; exploratory.',
 'real_model_validation':'First calibration forward per mode checks recomputed last row against full eager attention; retain errors and official scores.'}
(root/'configs/P007_lad_bridge.json').write_text(json.dumps(cfg,indent=2))
print(json.dumps({'calibration':[r['id'] for r in cal],'evaluation':[r['id'] for r in evaluation],'cal_n':len(cal),'eval_n':len(evaluation)}))

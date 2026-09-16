import json,hashlib
from pathlib import Path
import numpy as np
cfg=json.loads(Path('configs/P014_image_isolation.json').read_text())
p=Path('runs/P011_geometry_transport/predictions.jsonl')
assert hashlib.sha256(p.read_bytes()).hexdigest()==cfg['reference_predictions_sha256']
references=list(map(json.loads,p.read_text().splitlines()))
rs=list(map(json.loads,Path('runs/P014_image_isolation/predictions.jsonl').read_text().splitlines()))
assert len(rs)==cfg['n_records'] and len({r['id'] for r in rs})==len(rs)
by={m:{r['sample_id']:r for r in rs if r['mode']==m} for m in cfg['modes']}
for m in ['baseline','baseline_labeled']:
 by[m]={r['sample_id']:r for r in references if r['mode']==m}
assert all(len(v)==cfg['n_samples'] and v.keys()==by['baseline'].keys() for v in by.values())
clusters=sorted(set(cfg['image_clusters'].values()))
def ci(pairs):
 a=np.array([[sum(int(r['correct'])-int(b['correct']) for r,b in pairs if cfg['image_clusters'][r['base_id']]==c),
              sum(cfg['image_clusters'][r['base_id']]==c for r,b in pairs)] for c in clusters],dtype=float)
 rng=np.random.default_rng(17014)
 x=a[rng.integers(0,len(a),(20000,len(a)))].sum(1)
 return (np.quantile(x[:,0]/x[:,1],[.025,.975])*100).tolist()
report={'complete':True,'n_new_records':len(rs),'n_per_mode':cfg['n_samples'],'n_image_clusters':len(clusters),
 'modes':{},'comparisons':{},'limitations':cfg['limitations']+['Confidence intervals are exploratory and not adjusted for multiple comparisons.']}
for mode,rr in by.items():
 groups={}
 for r in rr.values():groups.setdefault((r['base_id'],tuple(r['image_order'])),[]).append(r)
 report['modes'][mode]={'correct':sum(r['correct'] for r in rr.values()),'n':len(rr),
 'invalid':sum('predicted_image_position' not in r for r in rr.values()),
 'by_question':{q:sum(r['correct'] for r in rr.values() if r['base_id']==q) for q in cfg['factorial']['ids']},
 'option_order_consistent_groups':sum(len({r.get('predicted_original_image',-1) for r in group})==1 and all('predicted_original_image' in r for r in group) for group in groups.values()),
 'n_option_groups':len(groups),'seconds':sum(r['seconds'] for r in rr.values()),
 'peak_memory_gb':max(r['peak_memory_gb'] for r in rr.values())}
for m in cfg['modes']:
 for reference in ['baseline','baseline_labeled','full_mask','isolated','isolated_labeled']:
  if m==reference:continue
  pairs=[(r,by[reference][k]) for k,r in by[m].items()]
  report['comparisons'][m+' vs '+reference]={
   'gains':sum(r['correct'] and not b['correct'] for r,b in pairs),
   'losses':sum(not r['correct'] and b['correct'] for r,b in pairs),
   'same_raw':sum(r['raw_prediction']==b['raw_prediction'] for r,b in pairs),
   'delta_95ci_pp':ci(pairs)}
pairs=[(r,by['baseline'][k]) for k,r in by['full_mask'].items()]
report['mask_numerical_audit']={
 'same_raw':sum(r['raw_prediction']==b['raw_prediction'] for r,b in pairs),'n':len(pairs),
 'max_candidate_score_error':max(abs(x-y) for r,b in pairs for fmt in r['candidate_scores'] for x,y in zip(r['candidate_scores'][fmt],b['candidate_scores'][fmt]))}
report['advance_screen']={m:all(report['modes'][m]['correct']>report['modes'][b]['correct'] for b in ['baseline','baseline_labeled','full_mask','isolated_labeled']) for m in ['template','skeleton']}
Path('results/P014_image_isolation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))

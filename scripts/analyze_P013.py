import json
from pathlib import Path
from collections import Counter
import numpy as np
cfg=json.loads(Path('configs/P013_cache_paths.json').read_text())
rows=list(map(json.loads,Path('runs/P013_cache_paths/predictions.jsonl').read_text().splitlines()))
assert len(rows)==cfg['n_records'] and len({r['id'] for r in rows})==len(rows)
by={c['id']:{r['sample_id']:r for r in rows if r['mode']==c['id']} for c in cfg['conditions']}
clusters=sorted(set(cfg['image_clusters'].values()))
def interval(data):
 a=np.array([[sum(n for c,n,d in data if c==g),sum(d for c,n,d in data if c==g)] for g in clusters])
 rng=np.random.default_rng(17013);z=a[rng.integers(0,len(a),(20000,len(a)))].sum(1)
 return (np.quantile(z[:,0]/np.maximum(z[:,1],1),[.025,.975])*100).tolist()
report={'complete':True,'n_records':len(rows),'n_per_mode':cfg['n_samples'],
 'modes':{},'transport':{},'integrity':{},'limitations':cfg['limitations']}
base=by['baseline_cached']
for mode,rr in by.items():
 pairs=[(r,base[k]) for k,r in rr.items()]
 report['modes'][mode]={'n':len(rr),'correct':sum(r['correct'] for r in rr.values()),
 'gains_vs_baseline':sum(r['correct'] and not b['correct'] for r,b in pairs),
 'losses_vs_baseline':sum(not r['correct'] and b['correct'] for r,b in pairs),
 'same_image_position_as_baseline':sum(r.get('predicted_image_position',-1)==b.get('predicted_image_position',-2) for r,b in pairs),
 'invalid':sum('predicted_image_position' not in r for r in rr.values())}
for part in ['full','start_only','other_prefix']:
 for shift in [1,3]:
  rr=by[part+':'+str(shift)];same=by[part+':0'];full=by['full:'+str(shift)];full0=by['full:0']
  pairs=[(r,rr[k]) for k,r in same.items()]
  valid=[(r,s) for r,s in pairs if 'predicted_image_position' in r and 'predicted_image_position' in s]
  def delta(r,s):return (s['predicted_image_position']-r['predicted_image_position'])%4
  hist=Counter(delta(r,s) for r,s in valid)
  aligned=[(r,full[k]) for k,r in rr.items() if 'predicted_image_position' in full[k] and 'predicted_image_position' in full0[k]
    and full[k]['predicted_image_position']!=full0[k]['predicted_image_position']]
  data=[(cfg['image_clusters'][r['base_id']],int(delta(r,s)==shift)-int(delta(r,s)==(-shift)%4),1) for r,s in valid]
  report['transport'][part+':'+str(shift)]={'n_pairs':len(pairs),'valid_pairs':len(valid),
   'shift_histogram_mod4':dict(hist),'expected_count':hist[shift],'opposite_count':hist[(-shift)%4],
   'expected_minus_opposite_cluster_95ci_pp':interval(data),
   'agreement_with_full_on_full_changed_cases':sum(r.get('predicted_image_position',-1)==s['predicted_image_position'] for r,s in aligned),
   'n_full_changed_cases':len(aligned)}
for mode in ['baseline_cached','full:0','full:1','full:3']:
 rr=list(by[mode].values());errors=[r['reference_P011_candidate_max_error'] for r in rr]
 report['integrity'][mode]={'n':len(rr),'same_raw_as_P011':sum(r['reference_P011_raw_match'] for r in rr),
 'candidate_score_max_error':max(errors),'candidate_score_p95_error':float(np.quantile(errors,.95))}
Path('results/P013_cache_paths.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))

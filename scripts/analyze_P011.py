import json
from collections import Counter
from pathlib import Path
import numpy as np
cfg=json.loads(Path('configs/P011_geometry_transport.json').read_text())
rs=list(map(json.loads,Path('runs/P011_geometry_transport/predictions.jsonl').read_text().splitlines()))
assert len(rs)==cfg['n_records'] and len({r['id'] for r in rs})==len(rs)
by={m:{r['sample_id']:r for r in rs if r['mode']==m} for m in cfg['modes']}
clusters=sorted(set(cfg['image_clusters'].values()))
def boot_ci(records):
 # Cluster resampling, preserving numerator/denominator and shared-image questions.
 a=np.array([[sum(num for group,num,den in records if group==c),
              sum(den for group,num,den in records if group==c)] for c in clusters],dtype=float)
 rng=np.random.default_rng(17011);sample=a[rng.integers(0,len(a),(20000,len(a)))].sum(1)
 val=sample[:,0]/np.maximum(sample[:,1],1)
 return (np.quantile(val,[.025,.975])*100).tolist()
report={'complete':True,'n_records':len(rs),'n_base_questions':cfg['n_base_questions'],
 'n_image_clusters':cfg['n_image_clusters'],'n_per_mode':cfg['n_samples'],'modes':{},'transport':{},
 'limitations':['Exploratory nine-cluster intervals','All questions appeared in earlier baseline evaluations; new only to this intervention',
 'Square donor changes visual features and positions jointly; cannot isolate every positional signal',
 'No Muir heldout use; no novel method or SOTA claim']}
base=by['baseline']
for mode,rr in by.items():
 pairs=[(r,base[k]) for k,r in rr.items()]
 z={'correct':sum(r['correct'] for r in rr.values()),'n':len(rr),
 'gains_vs_baseline':sum(r['correct'] and not b['correct'] for r,b in pairs),
 'losses_vs_baseline':sum(not r['correct'] and b['correct'] for r,b in pairs),
 'raw_same_as_baseline':sum(r['raw_prediction']==b['raw_prediction'] for r,b in pairs),
 'invalid':sum('predicted_image_position' not in r for r in rr.values()),
 'by_question':{q:sum(r['correct'] for r in rr.values() if r['base_id']==q) for q in cfg['factorial']['ids']}}
 z['cluster_bootstrap_delta_vs_baseline_95ci_pp']=boot_ci([(cfg['image_clusters'][r['base_id']],int(r['correct'])-int(b['correct']),1) for r,b in pairs])
 groups={}
 for r in rr.values():groups.setdefault((r['base_id'],tuple(r['image_order'])),[]).append(r)
 z['option_order_consistent_groups']=sum(len({r.get('predicted_original_image',-1) for r in group})==1 and all('predicted_original_image' in r for r in group) for group in groups.values())
 z['n_option_groups']=len(groups)
 report['modes'][mode]=z
for geometry in cfg['geometries']:
 for shift in [1,3]:
  a=by[geometry+':0'];b=by[geometry+':'+str(shift)]
  pairs=[(r,b[k]) for k,r in a.items()]
  valid=[(r,s) for r,s in pairs if 'predicted_image_position' in r and 'predicted_image_position' in s]
  def delta(r,s):return (s['predicted_image_position']-r['predicted_image_position'])%4
  hist=Counter(delta(r,s) for r,s in valid)
  opposite=(-shift)%4
  contributions=[(cfg['image_clusters'][r['base_id']],int(delta(r,s)==shift)-int(delta(r,s)==opposite),1) for r,s in valid]
  report['transport'][geometry+':'+str(shift)]={'n_pairs':len(pairs),'valid_pairs':len(valid),
   'shift_histogram_mod4':dict(hist),'expected_shift':shift,'opposite_shift':opposite,
   'expected_count':hist[shift],'opposite_count':hist[opposite],
   'expected_minus_opposite_rate_pp':100*(hist[shift]-hist[opposite])/max(len(valid),1),
   'cluster_bootstrap_95ci_pp':boot_ci(contributions),
   'by_question':{q:dict(Counter(delta(r,s) for r,s in valid if r['base_id']==q)) for q in cfg['factorial']['ids']}}
report['geometry_audit']={g:{
 'same_start_sequence_positions':sum(r['target_coordinates']['sequence'][0]==r['donor_coordinates']['sequence'][0] for r in by[g+':0'].values()),
 'same_image_grids':sum(r['target_coordinates']['grid']==r['donor_coordinates']['grid'] for r in by[g+':0'].values()),
 'n':len(by[g+':0'])} for g in cfg['geometries']}
Path('results/P011_geometry_transport.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))

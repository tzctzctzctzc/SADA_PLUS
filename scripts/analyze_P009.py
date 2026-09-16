import json
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
cfg=json.loads(Path('configs/P009_boundary_restore.json').read_text())
root=Path('runs/P009_boundary_restore')
assert json.loads((root/'summary.json').read_text())['complete']
rows=[json.loads(l) for l in (root/'predictions.jsonl').read_text().splitlines()]
data={m['id']:{r['sample_id']:r for r in rows if r['mode']==m['id']} for m in cfg['conditions']}
ids=list(data['baseline']);assert all(set(d)==set(ids) for d in data.values())
base_ids=cfg['factorial']['ids']
report={'scope':cfg['purpose'],'n_base_questions':len(base_ids),'n_per_mode':len(ids),'modes':{},'donor_position_contrasts':{},
 'limitations':cfg['limitations']+['Exploratory question-bootstrap intervals; only eight questions and five shared donor templates'],
 'integrity':{'oracle_max_score_error':max(r['oracle_score_max_error'] for r in data['clean_both'].values()),
 'oracle_all_generated_match':all(r['oracle_generation_match'] for r in data['clean_both'].values())}}
for mode,d in data.items():
    groups=defaultdict(list)
    for r in d.values():groups[(r['base_id'],tuple(r['image_order']))].append(r)
    delta=np.array([np.mean([int(r['correct'])-int(data['scale'][i]['correct']) for i,r in d.items() if r['base_id']==b]) for b in base_ids])
    rng=np.random.default_rng(20260917)
    boot=delta[rng.integers(0,len(delta),size=(10000,len(delta)))].mean(1)
    z={'correct':sum(r['correct'] for r in d.values()),'accuracy':float(np.mean([r['correct'] for r in d.values()])),
       'versus_scale_gains':sum(r['correct'] and not data['scale'][i]['correct'] for i,r in d.items()),
       'versus_scale_losses':sum(not r['correct'] and data['scale'][i]['correct'] for i,r in d.items()),
       'delta_vs_scale_pp':float(100*delta.mean()),'question_bootstrap_delta_vs_scale_95ci_pp':(100*np.quantile(boot,[.025,.975])).tolist(),
       'question_correct':{b:sum(r['correct'] for r in d.values() if r['base_id']==b) for b in base_ids},
       'image_position_counts':dict(Counter(r.get('predicted_image_position',-1) for r in d.values())),
       'option_reordering_consistent_groups':sum(len(v)==4 and all('predicted_original_image' in r for r in v) and len({r['predicted_original_image'] for r in v})==1 for v in groups.values()),
       'groups':len(groups)}
    report['modes'][mode]=z
for kind in ['start','end','both']:
    same=data['donor_'+kind];shift=data['donor_shift_'+kind]
    report['donor_position_contrasts'][kind]={'same_correct':report['modes']['donor_'+kind]['correct'],
      'shifted_correct':report['modes']['donor_shift_'+kind]['correct'],
      'same_minus_shifted_question_delta':{b:sum(int(same[i]['correct'])-int(shift[i]['correct']) for i in ids if same[i]['base_id']==b) for b in base_ids}}
for mode in ['baseline','scale']:
    old={r['id']:r for r in map(json.loads,Path('runs/P006_permutation_'+mode+'/predictions.jsonl').read_text().splitlines())}
    report['integrity'][mode+'_same_as_P006']=sum(r['raw_prediction']==old[i]['raw_prediction'] for i,r in data[mode].items())
Path('results/P009_boundary_restore.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))

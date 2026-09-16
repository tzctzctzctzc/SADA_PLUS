import json
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
cfg=json.loads(Path('configs/P006_permutation_factorial.json').read_text())
modes=['baseline','scale','canonical_start']
data={}
for mode in modes:
    p=Path('runs/P006_permutation_'+mode)
    assert json.loads((p/'summary.json').read_text())['complete'], mode
    rows=[json.loads(l) for l in (p/'predictions.jsonl').read_text().splitlines()]
    data[mode]={r['id']:r for r in rows}
ids=list(data['baseline'])
assert all(set(v)==set(ids) for v in data.values())
report={'n_questions':len(cfg['ids']),'n_permutations_per_mode':len(ids),'modes':{},
    'limitations':['Exploratory Mantis subset; 16 permutations are not independent samples',
    'Conditional token scores include repetition penalty and omit formats not represented by one token',
    'Image-position bias is a behavioral observation, not proof of causal ordinal encoding']}
for mode,d in data.items():
    rows=list(d.values());valid=[r for r in rows if 'predicted_image_position' in r]
    byimage=defaultdict(list);byoption=defaultdict(list);bybase=defaultdict(list)
    for r in valid:
        byimage[(r['base_id'],tuple(r['image_order']))].append(r)
        byoption[(r['base_id'],tuple(r['option_order']))].append(r)
        bybase[r['base_id']].append(r)
    z={'correct':sum(r['correct'] for r in rows),'accuracy':float(np.mean([r['correct'] for r in rows])),
       'valid':len(valid),'image_position_counts_0based':dict(Counter(r['predicted_image_position'] for r in valid)),
       'letter_counts':dict(Counter(r['parsed_prediction'] for r in valid)),
       'raw_prefix_counts':dict(Counter(r['raw_prediction'][:1] for r in rows)),
       'all16_correct_questions':sum(len(v)==16 and all(r['correct'] for r in v) for v in bybase.values()),
       'option_reordering_image_consistent_groups':sum(len(v)==4 and len(set(r['predicted_original_image'] for r in v))==1 for v in byimage.values()),
       'image_reordering_identity_consistent_groups':sum(len(v)==4 and len(set(r['predicted_original_image'] for r in v))==1 for v in byoption.values()),
       'groups_per_consistency_measure':len(byimage),'by_question':{},'format_conditions':{}}
    for bid,v in bybase.items():
        z['by_question'][bid]={'correct':sum(r['correct'] for r in v),'position_counts':dict(Counter(r['predicted_image_position'] for r in v)),
            'letter_counts':dict(Counter(r['parsed_prediction'] for r in v))}
    formats=sorted(set(k for r in rows for k in r['format_conditional_scores']))
    for fmt in formats:
        rr=[r for r in rows if fmt in r['format_conditional_scores']]
        pp=[r['option_order'][ord(r['format_conditional_scores'][fmt]['best_letter'])-65] for r in rr]
        ll=[r['format_conditional_scores'][fmt]['best_letter'] for r in rr]
        correct=[r['image_order'][pos]==r['gold_original_image'] for r,pos in zip(rr,pp)]
        z['format_conditions'][fmt]={'n':len(rr),'correct':sum(correct),'position_counts':dict(Counter(pp)),'letter_counts':dict(Counter(ll))}
    report['modes'][mode]=z
for mode in modes[1:]:
    pairs=[(data['baseline'][i],data[mode][i]) for i in ids if 'predicted_image_position' in data['baseline'][i] and 'predicted_image_position' in data[mode][i]]
    direction=[int(np.sign(b['predicted_image_position']-a['predicted_image_position'])) for a,b in pairs]
    report[mode+'_versus_baseline']={'current_position_shift':dict(Counter(direction)),
        'n_valid_position_pairs':len(pairs),'all_sample_gains':sum(not data['baseline'][i]['correct'] and data[mode][i]['correct'] for i in ids),'all_sample_losses':sum(data['baseline'][i]['correct'] and not data[mode][i]['correct'] for i in ids),'gains':sum(not a['correct'] and b['correct'] for a,b in pairs),'losses':sum(a['correct'] and not b['correct'] for a,b in pairs),
        'per_question_mean_position_shift':{bid:float(np.mean([b['predicted_image_position']-a['predicted_image_position'] for a,b in pairs if a['base_id']==bid])) for bid in cfg['ids']}}
Path('results/P006_factorial_analysis.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:({m:{a:b for a,b in v.items() if a!='by_question'} for m,v in val.items()} if k=='modes' else val) for k,val in report.items()},indent=2))

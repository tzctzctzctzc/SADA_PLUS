import json
from pathlib import Path
import numpy as np
cfg=json.loads(Path('configs/P008_protocol_cross.json').read_text())
records=[]
for size in ['3B','7B']:
    root=Path('runs/P008_protocol_'+size)
    assert json.loads((root/'summary.json').read_text())['complete']
    records.extend(map(json.loads,(root/'predictions.jsonl').read_text().splitlines()))
report={'scope':'Exploratory crossed protocol diagnostic on eight reused Mantis questions; permutations dependent',
 'n_records':len(records),'cells':[],'protocol_audits':{},'limitations':['Model, preprocessing, prompt interactions are behavioral evidence, not causal localization',
 'Square resizing changes aspect ratio and detail, not just token count','LAD and Mantis prompts differ in instructions as well as answer symbols']}
for model in cfg['models']:
 for res in cfg['preprocessing']:
  for prompt in cfg['prompts']:
    rr=[r for r in records if r['model']==model and r['preprocessing']==res and r['prompt']==prompt]
    a={(r['base_id'],tuple(r['image_order'])):r for r in rr if r['mode']=='baseline'}
    b={(r['base_id'],tuple(r['image_order'])):r for r in rr if r['mode']=='scale'}
    assert a.keys()==b.keys() and len(a)==32
    cell={'model':model,'preprocessing':res,'prompt':prompt,'n':32,'modes':{},
       'gains':sum(not a[i]['correct'] and b[i]['correct'] for i in a),
       'losses':sum(a[i]['correct'] and not b[i]['correct'] for i in a),
       'delta_pp':100*np.mean([int(b[i]['correct'])-int(a[i]['correct']) for i in a]),
       'question_delta':{base:float(np.mean([int(b[i]['correct'])-int(a[i]['correct']) for i in a if i[0]==base])) for base in cfg['ids']}}
    for mode,dd in [('baseline',a),('scale',b)]:
        correct=sum(r['correct'] for r in dd.values());boundary=[]
        for r in dd.values():
            att=np.array(r['attention']);parts=np.array(r['attention_parts'])
            top=np.argsort(att.mean(1))[-2:]
            boundary.append(parts[top][:,[0,2],:].sum()/att[top].sum())
        cell['modes'][mode]={'correct':correct,'accuracy':correct/32,
          'prediction_counts':{str(i):sum(r['prediction']==i for r in dd.values()) for i in range(-1,4)},
          'all4_correct_questions':sum(all(r['correct'] for r in dd.values() if r['base_id']==base) for base in cfg['ids']),
          'mean_top2_boundary_fraction':float(np.mean(boundary))}
    report['cells'].append(cell)
# Existing fully matched prompt/preprocessing cells provide an independent implementation audit.
for mode in ['baseline','scale']:
 old=[json.loads(l) for l in Path('runs/P006_permutation_'+mode+'/predictions.jsonl').read_text().splitlines()]
 old={(r['base_id'],tuple(r['image_order'])):r for r in old if r['option_order']==[0,1,2,3]}
 new=[r for r in records if r['model'].endswith('7B-Instruct') and r['preprocessing']=='native2048' and r['prompt']=='mantis_letters' and r['mode']==mode]
 report['protocol_audits']['P006_'+mode]={'n':len(new),'same_input_token_count':sum(r['input_tokens']==old[(r['base_id'],tuple(r['image_order']))]['input_tokens'] for r in new),
   'same_generated_answer':sum(r['raw']==old[(r['base_id'],tuple(r['image_order']))]['raw_prediction'] for r in new)}
Path('results/P008_protocol_cross.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))

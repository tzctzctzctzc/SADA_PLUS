import json, ast, re
from types import SimpleNamespace
from pathlib import Path
import numpy as np
cfg=json.loads(Path('configs/P008_protocol_cross.json').read_text())
records=[]
for size in ['3B','7B']:
    root=Path('runs/P008_protocol_'+size)
    assert json.loads((root/'summary.json').read_text())['complete']
    records.extend(map(json.loads,(root/'predictions.jsonl').read_text().splitlines()))
# Score saved text with the frozen P006 Mantis parser to preserve protocol fidelity.
source=Path('src/run_permutation_probe.py')
nodes=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in {'parse_answer','score'}]
namespace={'re':re,'args':SimpleNamespace(dataset='mantis_permutation')}
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),namespace)
parser_corrections=0
for r in records:
 if r['prompt']=='mantis_letters':
    doc={'question_type':'multi-choice','answer':chr(65+r['gold']),'options':[f'({chr(65+i)}) Image {i+1}' for i in range(4)]}
    parsed,correct=namespace['score'](doc,r['raw'])
    pred=ord(parsed)-65 if isinstance(parsed,str) and parsed in ['A','B','C','D'] else -1
    parser_corrections+=int(correct!=r['correct'] or pred!=r['prediction'])
    r['correct']=correct;r['prediction']=pred
report={'primary_mantis_parser':'Frozen P006 score/parse_answer functions applied offline to saved generation text','parser_corrections':parser_corrections,'scope':'Exploratory crossed protocol diagnostic on eight reused Mantis questions; permutations dependent',
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
for mode in ['baseline','scale']:
 old=[json.loads(l) for l in Path('runs/P007_lad_'+mode+'/predictions.jsonl').read_text().splitlines()]
 old={(r['base_id'],tuple(r['image_order'])):r for r in old}
 new=[r for r in records if r['model'].endswith('3B-Instruct') and r['preprocessing']=='square336' and r['prompt']=='lad_numbers' and r['mode']==mode]
 differences=[float(np.max(np.abs(np.array(r['candidate_scores']["''"])-old[(r['base_id'],tuple(r['image_order']))]['scores']))) for r in new]
 report['protocol_audits']['P007_'+mode]={'n':len(new),'same_candidate_argmax':sum(int(np.argmax(r['candidate_scores']["''"]))==old[(r['base_id'],tuple(r['image_order']))]['candidate_argmax'] for r in new),
   'same_input_token_count':sum(r['input_tokens']==old[(r['base_id'],tuple(r['image_order']))]['input_tokens'] for r in new),
   'max_candidate_score_difference':max(differences),'limitation':'P007 eager vs P008 SDPA BF16; numerical backend difference remains. P008 uses one backend for all paired cells.'}
Path('results/P008_protocol_cross.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))

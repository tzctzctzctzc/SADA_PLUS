import json,re
from pathlib import Path
from datasets import load_dataset
B={r['id']:r for r in map(json.loads,Path('runs/P005_muir_dev_baseline/predictions.jsonl').read_text().splitlines())}
S={r['id']:r for r in map(json.loads,Path('runs/P005_muir_dev_scale/predictions.jsonl').read_text().splitlines())}
ds=load_dataset('parquet',data_files={'test':[str(p) for p in Path('data/MUIRBENCH').rglob('*.parquet')]},split='test')
docs=ds.select([i for i,key in enumerate(ds['idx']) if key in B]).remove_columns('image_list')
rows=[];unmatched_none=[]
for d in docs:
 none=[j for j,v in enumerate(d['options']) if v.strip().casefold()=='none of the choices provided']
 assert len(none)==1,(d['idx'],d['options'])
 gold=ord(d['answer'].strip().upper())-65
 r={'id':d['idx'],'task':d['task'],'none_index':none[0],'gold_none':gold==none[0]}
 for mode,preds in [('baseline',B),('scale',S)]:
  pred=preds[d['idx']]['parsed_prediction']
  pi=ord(pred)-65 if isinstance(pred,str) and re.fullmatch('[A-Z]',pred) else -1
  r[mode]={'predicted_none':pi==none[0],'valid_letter':0<=pi<len(d['options']),'correct':preds[d['idx']]['correct']}
 rows.append(r)
report={'scope':'P005 Muir dev only, no heldout; diagnostic breakdown under original fixed prompt protocol','groups':{}}
for task in ['ALL']+sorted({r['task'] for r in rows}):
 rr=[r for r in rows if task=='ALL' or r['task']==task]
 z={'n':len(rr),'gold_none':sum(r['gold_none'] for r in rr)}
 for mode in ['baseline','scale']:
  z[mode]={'correct':sum(r[mode]['correct'] for r in rr),'predicted_none':sum(r[mode]['predicted_none'] for r in rr),
   'none_correct':sum(r['gold_none'] and r[mode]['correct'] for r in rr),
   'answerable_correct':sum(not r['gold_none'] and r[mode]['correct'] for r in rr),
   'invalid_letter':sum(not r[mode]['valid_letter'] for r in rr)}
 report['groups'][task]=z
Path('results/P005_answerability_analysis.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:report['groups'][k] for k in ['ALL','Visual Retrieval']},indent=2))

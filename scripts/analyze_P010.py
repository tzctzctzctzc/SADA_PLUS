import json,re
from pathlib import Path
from datasets import load_dataset
cfg=json.loads(Path('configs/P010_muir_protocol.json').read_text())
preds=list(map(json.loads,Path('runs/P010_muir_protocol/predictions.jsonl').read_text().splitlines()))
assert len(preds)==len(cfg['rows'])*4 and len({p['id'] for p in preds})==len(preds)
D=load_dataset('parquet',data_files={'test':sorted(str(p) for p in Path('data/MUIRBENCH').rglob('*.parquet'))},split='test')
allowed={r['id'] for r in cfg['rows']}
D=D.select([i for i,key in enumerate(D['idx']) if key in allowed]).remove_columns('image_list')
docs={d['idx']:d for d in D}
cells={(p,m):{r['sample_id']:r for r in preds if r['protocol']==p and r['mode']==m} for p in cfg['protocols'] for m in cfg['modes']}
for m,path in cfg['reference_predictions'].items():
 cells['original',m]={r['id']:r for r in map(json.loads,Path(path).read_text().splitlines()) if r['id'] in allowed}
def aggregate(ids,records):
 z={'n':len(ids),'correct':0,'gold_none':0,'none_correct':0,'answerable_correct':0,'predicted_none':0,'invalid_letter':0}
 for key in ids:
  d=docs[key];r=records[key]
  ni=[i for i,v in enumerate(d['options']) if v.strip().casefold()=='none of the choices provided']
  assert len(ni)==1
  gn=d['answer'].strip().upper()==chr(65+ni[0])
  pred=r['parsed_prediction'].strip().upper()
  z['correct']+=r['correct'];z['gold_none']+=gn
  z['none_correct']+=gn and r['correct'];z['answerable_correct']+=(not gn) and r['correct']
  z['predicted_none']+=pred==chr(65+ni[0])
  z['invalid_letter']+=not (len(pred)==1 and 'A'<=pred<chr(65+len(d['options'])))
 return z
report={'complete':True,'n_records':len(preds),'n_unique_questions':len(docs),'heldout_used':False,
 'scope':'Descriptive paired protocol audit on selected dev rows; reused images and counterparts are not independent trials','groups':{},'paired':{}}
for task in ['ALL']+sorted({d['task'] for d in docs.values()}):
 ids=sorted(k for k,d in docs.items() if task=='ALL' or d['task']==task)
 report['groups'][task]={p+':'+m:aggregate(ids,rec) for (p,m),rec in cells.items()}
for p in ['original']+cfg['protocols']:
 a=cells[p,'baseline'];b=cells[p,'scale']
 report['paired'][p+':scale_vs_baseline']={'gains':sum(not a[k]['correct'] and b[k]['correct'] for k in docs),
 'losses':sum(a[k]['correct'] and not b[k]['correct'] for k in docs)}
for p in cfg['protocols']:
 for m in cfg['modes']:
  a=cells['original',m];b=cells[p,m]
  report['paired'][p+':'+m+'_vs_original']={'gains':sum(not a[k]['correct'] and b[k]['correct'] for k in docs),
  'losses':sum(a[k]['correct'] and not b[k]['correct'] for k in docs)}
Path('results/P010_muir_protocol.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))

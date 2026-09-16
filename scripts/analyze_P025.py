"""Paired ImageCoDe analysis for the fixed pointwise controls."""
import json,collections
from pathlib import Path
import numpy as np
cfg=json.loads(Path('configs/P025_pointwise_imagecode.json').read_text())
data=json.loads(Path(cfg['data_records']).read_text());ev=[r for r in data if r['split']=='evaluation']
rows=list(map(json.loads,Path('runs/P025_pointwise_imagecode/predictions.jsonl').read_text().splitlines()))
assert len(rows)==cfg['expected_images'] and len({r['id'] for r in rows})==len(rows)
ref={r['id']:r for r in json.loads(Path('runs/P023_predictions_compared.json').read_text())}
assert set(ref)=={r['id'] for r in ev}
parent=list(range(len(ev)));seen={}
def root(i):
 while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
 return i
for i,r in enumerate(ev):
 for k in [r['cluster']]+[x['pixel_sha256'] for x in r['images']]:
  if k in seen:parent[root(i)]=root(seen[k])
  else:seen[k]=i
groups=collections.defaultdict(list)
for i in range(len(ev)):groups[root(i)].append(i)
blocks=list(groups.values());rng=np.random.default_rng(17025)
boot=[np.concatenate([blocks[j] for j in rng.integers(len(blocks),size=len(blocks))]) for _ in range(5000)]
byq=collections.defaultdict(list)
for r in rows:byq[r['base_id']].append(r)
gold=np.array([r['gold'] for r in ev]);methods={m:np.array([ref[r['id']]['predictions'][m] for r in ev]) for m in next(iter(ref.values()))['predictions']}
new_methods=['matching_score','caption_logprob_mean'];tie={}
for method in new_methods:
 scores=np.array([[r[method] for r in sorted(byq[q['id']],key=lambda r:r['image_index'])] for q in ev])
 assert scores.shape==(256,10)
 methods[method]=scores.argmax(1)
 tied=scores==scores.max(1,keepdims=True)
 tie[method]={'questions_with_top_ties':int((tied.sum(1)>1).sum()),'expected_correct_uniform_tie':float((tied[np.arange(len(ev)),gold]/tied.sum(1)).sum())}
correct={m:(v==gold).astype(int) for m,v in methods.items()}
report={'complete':True,'n_eval':len(ev),'n_clusters':len(blocks),'correct':{m:int(c.sum()) for m,c in correct.items()},'comparisons':{},'ties':tie,'by_kind':{},'limitations':cfg['limits']}
for method in new_methods:
 for baseline in ['raw_logits','patch_attention_calibrated','patch_density_calibrated','code_AGD_raw','SigLIP2']:
  d=correct[method]-correct[baseline]
  report['comparisons'][method+' minus '+baseline]={'difference_pp':float(d.mean()*100),'cluster_95ci_pp':np.percentile([d[ix].mean()*100 for ix in boot],[2.5,97.5]).tolist(),'gains':int((d>0).sum()),'losses':int((d<0).sum())}
for kind in ['static','video']:
 ix=np.array([i for i,r in enumerate(ev) if r['kind']==kind])
 report['by_kind'][kind]={'n':len(ix),'correct':{m:int(c[ix].sum()) for m,c in correct.items()}}
report['cases']={
 'patch_correct_pointwise_both_wrong':[ev[i]['id'] for i in range(len(ev)) if correct['patch_attention_calibrated'][i] and not correct['matching_score'][i] and not correct['caption_logprob_mean'][i]],
 'listwise_correct_caption_wrong':[ev[i]['id'] for i in range(len(ev)) if correct['raw_logits'][i] and not correct['caption_logprob_mean'][i]],
 'caption_correct_listwise_wrong':[ev[i]['id'] for i in range(len(ev)) if correct['caption_logprob_mean'][i] and not correct['raw_logits'][i]]}
Path('results/P025_pointwise_imagecode.json').write_text(json.dumps(report,indent=2))
Path('runs/P025_compared_predictions.json').write_text(json.dumps([{'id':r['id'],'gold':r['gold'],'cluster':root(i),'predictions':{m:int(v[i]) for m,v in methods.items()}} for i,r in enumerate(ev)],indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2))

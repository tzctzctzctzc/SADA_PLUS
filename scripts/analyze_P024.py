"""Paired comparison of pointwise controls with existing frozen baselines."""
import json,collections
from pathlib import Path
import numpy as np
cfg=json.loads(Path('configs/P024_pointwise.json').read_text());dc=json.loads(Path(cfg['data_config']).read_text())
rows=list(map(json.loads,Path('runs/P024_pointwise/predictions.jsonl').read_text().splitlines()))
assert len(rows)==96 and len({r['id'] for r in rows})==96
docs={x['id']:x for x in dc['evaluation']};parent={q:q for q in docs}
def root(q):
 while parent[q]!=q:parent[q]=parent[parent[q]];q=parent[q]
 return q
for q,d in docs.items():
 for k,e in docs.items():
  if set(d['image_hashes'])&set(e['image_hashes']):parent[root(q)]=root(k)
groups=sorted({root(q) for q in docs})
ref={r['id']:r for r in json.loads(Path('runs/P017_lad_predictions.json').read_text()) if r['mode']=='native2048'}
raw={r['id']:r for r in json.loads(Path('runs/P021_reconstructed_scores.json').read_text()) if r['tag']=='P017' and r['mode']=='native2048' and r['split']=='evaluation'}
sig={r['id']:r for r in map(json.loads,Path('runs/P020_siglip_retrieval/predictions.jsonl').read_text().splitlines()) if r['split']=='evaluation'}
methods=['matching_score','caption_logprob_mean'];records=[]
tie_report={}
for q,d in docs.items():
 rr=sorted([r for r in rows if r['base_id']==q],key=lambda r:r['image_index']);assert [r['image_index'] for r in rr]==list(range(4))
 gold=rr[0]['gold_original'];scores={m:np.array([r[m] for r in rr]) for m in methods}
 tie_report[q]={m:{'top_ties':int((s==s.max()).sum()),'expected_correct_uniform_tie':float(s[gold]==s.max())/int((s==s.max()).sum())} for m,s in scores.items()}
 for shift,perm in enumerate(dc['permutations']):
  idx=q+':p'+str(shift);g=perm.index(gold);assert g==ref[idx]['gold']==raw[idx]['gold']
  correct={m:int(np.argmax(s[perm]))==g for m,s in scores.items()}
  correct.update({'listwise_raw':raw[idx]['predictions']['restored_raw_logits']==g,'patch_calibrated':ref[idx]['predictions']['patch_attention_calibrated']==g,'code_AGD_raw':raw[idx]['predictions']['code_AGD_raw']==g,'SigLIP2':sig[idx]['correct']})
  records.append({'id':idx,'cluster':root(q),'correct':correct})
report={'complete':True,'n_questions':len(docs),'n_permutations':len(records),'n_image_clusters':len(groups),
 'correct':{m:sum(r['correct'][m] for r in records) for m in records[0]['correct']},
 'ties':{m:{'questions_with_top_ties':sum(t[m]['top_ties']>1 for t in tie_report.values()),'expected_correct_questions_uniform_tie':sum(t[m]['expected_correct_uniform_tie'] for t in tie_report.values())} for m in methods},
 'comparisons':{},'scope':cfg['scope'],'limitations':cfg['limits']+['Tie-breaking uses displayed candidate order, matching the listwise baseline; a uniform-tie expected accuracy is additionally reported.']}
for method in methods:
 for baseline in ['listwise_raw','patch_calibrated','code_AGD_raw','SigLIP2']:
  a=np.array([[sum(int(r['correct'][method])-int(r['correct'][baseline]) for r in records if r['cluster']==g),sum(r['cluster']==g for r in records)] for g in groups])
  rng=np.random.default_rng(17024);b=a[rng.integers(len(a),size=(20000,len(a)))].sum(1)
  report['comparisons'][method+' minus '+baseline]={'difference_pp':float(a[:,0].sum()/a[:,1].sum()*100),'cluster_95ci_pp':(np.quantile(b[:,0]/b[:,1],[.025,.975])*100).tolist()}
Path('results/P024_pointwise.json').write_text(json.dumps(report,indent=2))
Path('runs/P024_comparisons.json').write_text(json.dumps({'records':records,'ties':tie_report},indent=2))
print(json.dumps(report,indent=2))

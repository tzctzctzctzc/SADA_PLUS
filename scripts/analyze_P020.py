import json,hashlib
from pathlib import Path
import numpy as np
cfg=json.loads(Path('configs/P020_siglip_retrieval.json').read_text())
dc=json.loads(Path(cfg['data_config']).read_text())
rows=list(map(json.loads,Path('runs/P020_siglip_retrieval/predictions.jsonl').read_text().splitlines()))
assert len(rows)==96 and len({r['id'] for r in rows})==96
ev={r['id']:r for r in rows if r['split']=='evaluation'}
docs={d['id']:d for d in dc['evaluation']}
parent={q:q for q in docs}
def find(q):
 while parent[q]!=q:parent[q]=parent[parent[q]];q=parent[q]
 return q
for q,a in docs.items():
 for k,b in docs.items():
  if set(a['image_hashes'])&set(b['image_hashes']):parent[find(q)]=find(k)
clusters={q:find(q) for q in docs};groups=sorted(set(clusters.values()))
report={'complete':True,'n_eval_questions':len(docs),'n_eval_permutations':len(ev),'n_image_clusters':len(groups),
 'siglip_correct':sum(r['correct'] for r in ev.values()),'correct_unique_questions':sum(r['correct'] for r in ev.values())//4,
 'permutation_consistent_questions':sum(len({r['predicted_original_image'] for r in ev.values() if r['base_id']==q})==1 for q in docs),
 'reference_note':'Original P017 logits and AGD use processed scores with repetition_penalty=1.05; see P021 for corrected raw comparisons.',
 'comparisons':{},'limitations':['Same repeatedly used Mantis questions; no new-data generalization claim.',
 'Independent image encoding makes permutation invariance intrinsic; its four permutations are not four independent examples.',
 'Encoder uses official 384px preprocessing; VLM resolution protocols reported separately.']}
ref=list(map(lambda x:x,json.loads(Path('runs/P017_lad_predictions.json').read_text())))
for mode in ['square336','native2048']:
 for method in ['logits','agd_official','patch_attention_calibrated']:
  rr={r['id']:r for r in ref if r['mode']==mode};assert rr.keys()==ev.keys()
  records=[(clusters[r['base_id']],int(r['correct'])-int(rr[k]['predictions'][method]==rr[k]['gold']),1) for k,r in ev.items()]
  a=np.array([[sum(n for g,n,d in records if g==c),sum(d for g,n,d in records if g==c)] for c in groups])
  rng=np.random.default_rng(17020);b=a[rng.integers(0,len(a),(20000,len(a)))].sum(1)
  report['comparisons'][mode+':'+method]={'reference_correct':sum(r['predictions'][method]==r['gold'] for r in rr.values()),
   'siglip_minus_reference_cluster_95ci_pp':(np.quantile(b[:,0]/b[:,1],[.025,.975])*100).tolist(),
   'siglip_gains':sum(ev[k]['correct'] and r['predictions'][method]!=r['gold'] for k,r in rr.items()),
   'siglip_losses':sum(not ev[k]['correct'] and r['predictions'][method]==r['gold'] for k,r in rr.items())}
Path('results/P020_siglip_retrieval.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))

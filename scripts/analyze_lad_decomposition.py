"""Post-hoc algebra and calibration-sample uncertainty for existing LAD/AGD."""
import json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,'src')
from lad_bridge import official_lad_functions
F=official_lad_functions()
report={'scope':'Post-hoc baseline diagnosis; no new method or confirmatory selection',
 'identity':'For row-normalized p, L=E[z|Y] and I=b*11^T+k*Id, z-p*(L-I) equals z-p*L+k*p plus a class-constant b.',
 'implication':'AGD combines attention-based index evidence with residualized logits. The ideal target is an assumed symmetric mean-logit structure, not directly observed unbiased logits.',
 'bootstrap':'Resample five calibration questions as blocks, preserving their four cyclic permutations; evaluation fixed. Report sensitivity, not a generalization confidence interval.',
 'results':{}}
for tag in ['P007','P015']:
 cfg=json.loads(Path('configs/'+tag+'_lad_bridge.json').read_text())
 for mode in cfg['modes']:
  pth=Path('runs/'+tag+'_lad_'+mode+'/predictions.jsonl')
  rows=list(map(json.loads,pth.read_text().splitlines()))
  assert len(rows)==4*(len(cfg['calibration'])+len(cfg['evaluation']))
  cal=[r for r in rows if r['split']=='calibration'];ev=[r for r in rows if r['split']=='evaluation']
  z=np.array([r['scores'] for r in ev]);gold=np.array([r['gold'] for r in ev])
  def estimate(c):
   stats={i:[np.array(r['scores']) for r in c if r['gold']==i] for i in range(4)}
   B,I,L=F['compute_bias_matrix'](stats,4,verbose=False)
   prior=np.mean([r['layer_attention'] for r in c],axis=0)
   probs=np.array([F['sharpen_distribution'](F['select_attention_scores'](np.array(r['layer_attention']),prior,cfg['topk']),cfg['sharpen_power']) for r in ev])
   k=float(I[0,0]-I[0,1])
   official=z-probs@B
   algebra=z-probs@L+k*probs+I[0,1]
   return official,probs,L,k,float(np.max(np.abs(official-algebra)))
  official,probs,L,k,error=estimate(cal)
  assert error<1e-9
  variants={'logits':z,'attention_only':probs,'official_AGD':official,
   'attention_injection_only':z+k*probs,'conditional_residual_only':z-probs@L,
   'unconditional_centering':z-L.mean(0)}
  counts={name:int((scores.argmax(1)==gold).sum()) for name,scores in variants.items()}
  pred=official.argmax(1)
  ids=[r['id'] for r in cfg['calibration']]
  blocks={q:[r for r in cal if r['base_id']==q] for q in ids}
  rng=np.random.default_rng(17016);bc=[];ks=[];flip=[]
  for _ in range(1000):
   sampled=[r for q in rng.choice(ids,len(ids),replace=True) for r in blocks[q]]
   out,_,_,kk,ee=estimate(sampled)
   assert ee<1e-9
   pp=out.argmax(1);bc.append(int((pp==gold).sum()));ks.append(kk);flip.append(int((pp!=pred).sum()))
  report['results'][tag+':'+mode]={'n':len(ev),'correct':counts,'k':k,'algebra_max_error':error,
   'AGD_vs_injection_same_predictions':int((pred==variants['attention_injection_only'].argmax(1)).sum()),
   'calibration_bootstrap_1000':{'correct_quantiles_025_50_975':np.quantile(bc,[.025,.5,.975]).tolist(),
    'k_quantiles_025_50_975':np.quantile(ks,[.025,.5,.975]).tolist(),
    'prediction_change_count_quantiles_025_50_975':np.quantile(flip,[.025,.5,.975]).tolist()}}
Path('results/P016_lad_decomposition.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))

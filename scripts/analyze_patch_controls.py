"""Calibration sensitivity and a predeclared token-count control for patch readout."""
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,'src')
from lad_bridge import official_lad_functions
F=official_lad_functions()
parser=argparse.ArgumentParser();parser.add_argument('--tag',choices=['P015','P017'],required=True)
args=parser.parse_args();tag=args.tag
cfg=json.loads(Path('configs/'+('P017_lad_resolution' if tag=='P017' else 'P015_lad_bridge')+'.json').read_text())
spec=json.loads(Path('configs/P018_patch_controls.json').read_text())
report={'tag':tag,'scope':spec['scope'],'modes':{}}
for mode in cfg['modes']:
 rows=list(map(json.loads,Path('runs/'+tag+'_lad_'+mode+'/predictions.jsonl').read_text().splitlines()))
 assert len(rows)==96
 cal=[r for r in rows if r['split']=='calibration'];ev=[r for r in rows if r['split']=='evaluation']
 ids=[r['id'] for r in cfg['calibration']]
 assert all(not(set(a['image_hashes'])&set(b['image_hashes'])) for i,a in enumerate(cfg['calibration']) for b in cfg['calibration'][i+1:])
 cal_ix={q:[i for i,r in enumerate(cal) if r['base_id']==q] for q in ids}
 gold=np.array([r['gold'] for r in ev])
 def features(rr,kind):
  val=np.array([r['layer_attention'] if kind=='all_span' else r['parts']['patch'] for r in rr])
  if kind=='patch_density':
   counts=np.array([[np.prod(g)//4 for g in r['image_grid_thw']] for r in rr])
   val=val/counts[:,None,:]
  return val
 out={}
 rng=np.random.default_rng(spec['seed'])
 resamples=[np.concatenate([cal_ix[q] for q in rng.choice(ids,len(ids),replace=True)]) for _ in range(spec['n_resamples'])]
 for kind in ['all_span','patch','patch_density']:
  cf=features(cal,kind);ef=features(ev,kind)
  ix=np.argsort(ef.mean(2),axis=1)[:,-cfg['topk']:]
  num=np.take_along_axis(ef,ix[:,:,None],axis=1).mean(1)
  def predict(prior):
   den=np.exp(np.log(prior[ix]+1e-20).mean(1))
   return (num/den).argmax(1)
  prior=cf.mean(0);pred=predict(prior)
  direct=np.array([F['select_attention_scores'](a,prior,cfg['topk']).argmax() for a in ef])
  assert np.array_equal(pred,direct)
  counts=[];flips=[]
  for ci in resamples:
   pp=predict(cf[ci].mean(0));counts.append(int((pp==gold).sum()));flips.append(int((pp!=pred).sum()))
  out[kind]={'correct':int((pred==gold).sum()),'n':len(ev),'raw_correct':int((num.argmax(1)==gold).sum()),
   'calibration_correct_quantiles_025_50_975':np.quantile(counts,[.025,.5,.975]).tolist(),
   'prediction_change_quantiles_025_50_975':np.quantile(flips,[.025,.5,.975]).tolist(),
   'predictions':pred.tolist()}
 report['modes'][mode]=out
report['limitations']=['Calibration resampling with evaluation fixed is sensitivity, not a generalization confidence interval.',
 'Patch density is a simple count-normalization baseline, not a novelty claim.',
 'Modes and metrics are exploratory; no best-performing variant will be presented as preselected.']
Path('results/P018_patch_controls_'+tag+'.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({m:{k:{x:y for x,y in z.items() if x!='predictions'} for k,z in a.items()} for m,a in report['modes'].items()},indent=2))

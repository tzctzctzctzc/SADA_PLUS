"""Frozen paired analysis for the independent ImageCoDe pilot."""
import json,sys,hashlib,collections
from pathlib import Path
import numpy as np
from scipy.special import softmax
sys.path.insert(0,'src')
from lad_bridge import official_lad_functions
cfg=json.loads(Path('configs/P023_imagecode.json').read_text());F=official_lad_functions()
data=json.loads(Path('runs/P023_data/records.json').read_text())
vlm={r['id']:r for r in map(json.loads,Path('runs/P023_imagecode_vlm/predictions.jsonl').read_text().splitlines())}
sig={r['id']:r for r in map(json.loads,Path('runs/P023_imagecode_siglip/predictions.jsonl').read_text().splitlines())}
assert set(vlm)==set(sig)=={r['id'] for r in data}
rows=[vlm[r['id']] for r in data];n=cfg['n_images']
gold=np.array([r['gold'] for r in rows]);ci=np.array([i for i,r in enumerate(rows) if r['split']=='calibration']);ei=np.array([i for i,r in enumerate(rows) if r['split']=='evaluation'])
z=np.array([r['scores'] for r in rows]);a=np.array([r['layer_attention'] for r in rows]);patch=np.array([r['parts']['patch'] for r in rows]);tokens=np.array([r['image_tokens'] for r in rows])
density=patch/tokens[:,None,:]
def retrieve(x,selection=None,calibrated=True):
 prior=x[ci].mean(0);out=[]
 for i,item in enumerate(x):
  selected=np.argsort((item if selection is None else selection[i]).mean(1))[-cfg['topk']:]
  logscore=np.log(item[selected].mean(0)+1e-20)
  if calibrated:logscore-=np.log(prior[selected]+1e-20).mean(0)
  out.append(softmax(cfg['sharpen_power']*logscore))
 return np.array(out)
all_att=retrieve(a);patch_att=retrieve(patch);dense_att=retrieve(density)
B,I,L=F['compute_bias_matrix']({i:[x for x,g in zip(z[ci],gold[ci]) if g==i] for i in range(n)},n,verbose=False)
methods={'raw_logits':z,'all_span_attention_calibrated':all_att,'patch_attention_calibrated':patch_att,
 'patch_density_calibrated':dense_att,'patch_on_total_layers':retrieve(patch,selection=a),
 'code_AGD_raw':z-all_att@B,'SigLIP2':np.array([sig[r['id']]['scores'] for r in rows]),
 'all_span_attention_raw':retrieve(a,calibrated=False),'patch_attention_raw':retrieve(patch,calibrated=False)}
parent=list(range(len(data)))
def find(i):
 while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
 return i
seen={}
for i,row in enumerate(data):
 for key in [row['cluster']]+[im['pixel_sha256'] for im in row['images']]:
  if key in seen:parent[find(i)]=find(seen[key])
  else:seen[key]=i
assert not {find(i) for i in ci}&{find(i) for i in ei}
clusters=collections.defaultdict(list)
for k,i in enumerate(ei):clusters[find(i)].append(k)
blocks=list(clusters.values());rng=np.random.default_rng(23017)
bootstrap=[np.concatenate([blocks[x] for x in rng.integers(len(blocks),size=len(blocks))]) for _ in range(5000)]
correct={k:(v[ei].argmax(-1)==gold[ei]).astype(int) for k,v in methods.items()}
report={'complete':True,'n_calibration':len(ci),'n_evaluation':len(ei),'n_eval_clusters':len(blocks),
 'data_audit':json.loads(Path('runs/P023_data/audit.json').read_text()),'methods':{},'paired_comparisons':{},
 'protocol':'Explicit image labels A-J, repetition_penalty=1, original chronological order, SDPA BF16, native aspect ratio with max_pixels=448^2.',
 'limits':cfg['limits'],'config_sha256':hashlib.sha256(Path('configs/P023_imagecode.json').read_bytes()).hexdigest()}
for name,v in methods.items():
 order=np.argsort(-v[ei],axis=1,kind='stable')
 ranks=np.array([np.where(order[j]==gold[i])[0][0]+1 for j,i in enumerate(ei)])
 c=correct[name]
 report['methods'][name]={'correct':int(c.sum()),'accuracy':float(c.mean()),'mrr_stable_label_ties':float((1/ranks).mean()),
  'by_kind':{kind:{'n':sum(rows[i]['kind']==kind for i in ei),'correct':int(sum(c[j] for j,i in enumerate(ei) if rows[i]['kind']==kind))} for kind in ['static','video']}}
for name in ['patch_attention_calibrated','patch_density_calibrated','patch_on_total_layers']:
 for reference in ['raw_logits','all_span_attention_calibrated','code_AGD_raw','SigLIP2']:
  diff=correct[name]-correct[reference]
  report['paired_comparisons'][name+' minus '+reference]={'difference_pp':float(diff.mean()*100),
   'cluster_95ci_pp':np.percentile([diff[ix].mean()*100 for ix in bootstrap],[2.5,97.5]).tolist(),
   'gains':int((diff>0).sum()),'losses':int((diff<0).sum())}
report['image_length_audit']={'min_tokens':int(tokens.min()),'max_tokens':int(tokens.max()),'rows_with_unequal_image_tokens':int((tokens.max(1)!=tokens.min(1)).sum())}
report['siglip_truncated_eval_captions']=sum(sig[r['id']]['truncated'] for r in data if r['split']=='evaluation')
report['first_token_outside_label_set']=sum(r['generated_first_token'].strip() not in cfg['labels'] for r in rows if r['split']=='evaluation')
Path('results/P023_imagecode.json').write_text(json.dumps(report,indent=2))
details=[{'id':rows[i]['id'],'gold':int(gold[i]),'predictions':{k:int(v[i].argmax()) for k,v in methods.items()}} for i in ei]
Path('runs/P023_predictions_compared.json').write_text(json.dumps(details,indent=2))
print(json.dumps(report,indent=2))

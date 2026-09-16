"""Post-hoc audit: decode penalty and literal paper equations versus released LAD code."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.special import softmax,logsumexp
from transformers import AutoTokenizer
sys.path.insert(0,'src')
from lad_bridge import official_lad_functions
F=official_lad_functions();tokenizers={}
report={'scope':'Fixed recorded layers, prompts and data; formula-level comparison, not original benchmark reproduction.',
 'source_differences':{
  'conditional_statistics':'Paper Eq1 averages candidate probabilities; released compute_bias_matrix averages scores.',
  'layer_pooling':'Paper Eq11 averages log attention; released select_attention_scores takes log of mean attention for the numerator.',
  'bias_pooling':'Paper Eq13 is an arithmetic mixture of bias probabilities; released AGDMethod uses a weighted sum of bias scores.',
  'decoding':'Recorded generate scores include Qwen repetition_penalty=1.05. Raw scores reconstructed by inverting only this active processor.'},
 'raw_recovery_validation':'results/P022_raw_score_validation.json',
 'raw_recovery':'Inverse repetition processor then nearest BF16 value; preserves original logit ties.',
 'cases':{},'limitations':['Paper-style equations use the same recorded last-15-layer range and image spans; no claim of end-to-end paper reproduction.',
 'Eq5 bias weights are used without additional row renormalization, following the displayed equation.',
 'Recorded four single-token candidate scores are normalized over those four candidates for the probability variant.',
 'This is post-hoc baseline auditing, not a new method or confirmatory selection.']}
raw_records=[]
for tag,cn in [('P007','P007_lad_bridge'),('P015','P015_lad_bridge'),('P017','P017_lad_resolution')]:
 cfg=json.loads(Path('configs/'+cn+'.json').read_text())
 model=cfg['model']
 if model not in tokenizers:tokenizers[model]=AutoTokenizer.from_pretrained(Path('models')/model,local_files_only=True)
 tokenizer=tokenizers[model]
 docs={d['id']:d for d in cfg['calibration']+cfg['evaluation']}
 candidate_ids=[tokenizer.convert_tokens_to_ids(str(i)) for i in range(1,5)]
 seen={}
 for q,d in docs.items():
  prompt=(f"Given 4 images indexed from 1 to 4, identify the image that best matches the provided caption."
    f" Respond with the index number only and nothing else.\nCaption: {d['caption']}\nAnswer: ")
  messages=[{'role':'user','content':[*[{'type':'image'} for _ in range(4)],{'type':'text','text':prompt}]}]
  text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
  ids=set(tokenizer.encode(text,add_special_tokens=False));seen[q]=np.array([i in ids for i in candidate_ids])
 for mode in cfg['modes']:
  root=Path('runs/'+tag+'_lad_'+mode)
  meta=json.loads((root/'config.json').read_text())
  gc=meta['generation_config'];penalty=gc['repetition_penalty']
  assert gc['no_repeat_ngram_size']==0 and gc['bad_words_ids'] is None and gc['renormalize_logits']==False
  rows=list(map(json.loads,(root/'predictions.jsonl').read_text().splitlines()))
  assert len(rows)==96
  zz=np.array([r['scores'] for r in rows]);mask=np.stack([seen[r['base_id']] for r in rows])
  raw_unrounded=np.where(mask,np.where(zz>=0,zz*penalty,zz/penalty),zz)
  assert meta['dtype']=='torch.bfloat16'
  # lm_head logits are BF16 then cast to FP32 by generate; restore the original grid.
  # P022 validates exact reconstruction against output_logits on all six settings.
  raw=torch.from_numpy(raw_unrounded).to(torch.bfloat16).float().numpy()
  # Qwen visual placeholder expansion cannot introduce digit candidate tokens.
  ci=np.array([i for i,r in enumerate(rows) if r['split']=='calibration'])
  ei=np.array([i for i,r in enumerate(rows) if r['split']=='evaluation'])
  gold=np.array([r['gold'] for r in rows]);a=np.array([r['layer_attention'] for r in rows])
  prior=a[ci].mean(0);att=[];paper_att=[]
  for x in a:
   selected=np.argsort(x.mean(1))[-cfg['topk']:]
   att.append(F['sharpen_distribution'](F['select_attention_scores'](x,prior,cfg['topk']),cfg['sharpen_power']))
   logp=(np.log(x[selected]+1e-20)-np.log(prior[selected]+1e-20)).mean(0)
   paper_att.append(softmax(logp*cfg['sharpen_power']))
  att=np.array(att);paper_att=np.array(paper_att)
  variants={'processed_logits':zz,'restored_raw_logits':raw,'code_attention_only':att,'paper_attention_only':paper_att}
  details={}
  for label,z in [('processed',zz),('raw',raw)]:
   stats={i:[x for x,g in zip(z[ci],gold[ci]) if g==i] for i in range(4)}
   B,I,L=F['compute_bias_matrix'](stats,4,verbose=False)
   variants['code_AGD_'+label]=z-att@B
   C=np.stack([softmax(z[ci][gold[ci]==i],axis=-1).mean(0) for i in range(4)])
   logC=np.log(C)
   loggamma=max(logC[i,i]-logC[i,j] for i in range(4) for j in range(4) if i!=j)
   beta=logC-loggamma*np.eye(4)
   variants['paper_equations_'+label]=z-logsumexp(np.log(paper_att[:,:,None]+1e-300)+beta[None,:,:],axis=1)
   variants['code_only_pooling_replaced_'+label]=z-logsumexp(np.log(att[:,:,None]+1e-300)+B[None,:,:],axis=1)
   gap=logsumexp(np.log(att[:,:,None]+1e-300)+B[None,:,:],axis=1)-att@B
   assert gap.min()>-1e-10
   details[label]={'paper_loggamma':float(loggamma),'code_k':float(I[0,0]-I[0,1]),'pooling_Jensen_gap_max':float(gap.max())}
  counts={name:int((val[ei].argmax(1)==gold[ei]).sum()) for name,val in variants.items()}
  changed=zz[ei].argmax(1)!=raw[ei].argmax(1)
  report['cases'][tag+':'+mode]={'n':len(ei),'correct':counts,'penalty_changed_predictions':int(changed.sum()),
   'penalized_candidate_patterns':sorted({tuple(int(v) for v in seen[r['base_id']]) for r in rows}),
   'details':details}
  for i,r in enumerate(rows):
   raw_records.append({'tag':tag,'mode':mode,'id':r['id'],'split':r['split'],'penalized':mask[i].tolist(),
    'raw_reconstructed':raw[i].tolist(),'processed':zz[i].tolist(),'gold':r['gold'],
    'predictions':{name:int(val[i].argmax()) for name,val in variants.items()}})

# Existing P017 patch predictions versus corrected raw scores, grouped by image reuse.
cfg17=json.loads(Path('configs/P017_lad_resolution.json').read_text())
docs={d['id']:d for d in cfg17['evaluation']};parent={q:q for q in docs}
def root(q):
 while parent[q]!=q:parent[q]=parent[parent[q]];q=parent[q]
 return q
for q,a in docs.items():
 for k,b in docs.items():
  if set(a['image_hashes'])&set(b['image_hashes']):parent[root(q)]=root(k)
groups=sorted({root(q) for q in docs})
ref=json.loads(Path('runs/P017_lad_predictions.json').read_text())
report['patch_vs_corrected_raw']={}
for mode in ['square336','native2048']:
 rr={r['id']:r for r in ref if r['mode']==mode}
 raw={r['id']:r for r in raw_records if r['tag']=='P017' and r['mode']==mode and r['split']=='evaluation'}
 records=[(root(k.split(':p')[0]),int(r['predictions']['patch_attention_calibrated']==r['gold'])-int(raw[k]['predictions']['restored_raw_logits']==r['gold'])) for k,r in rr.items()]
 a=np.array([[sum(d for g,d in records if g==c),sum(g==c for g,d in records)] for c in groups])
 rng=np.random.default_rng(17021);b=a[rng.integers(len(a),size=(20000,len(a)))].sum(1)
 report['patch_vs_corrected_raw'][mode]={'n':len(records),'clusters':len(groups),'patch_correct':sum(r['predictions']['patch_attention_calibrated']==r['gold'] for r in rr.values()),
  'raw_correct':sum(r['predictions']['restored_raw_logits']==r['gold'] for r in raw.values()),
  'difference_pp':sum(d for g,d in records)/len(records)*100,
  'cluster_95ci_pp':(np.quantile(b[:,0]/b[:,1],[.025,.975])*100).tolist()}

Path('runs/P021_reconstructed_scores.json').write_text(json.dumps(raw_records,indent=2)+'\n')
Path('results/P021_lad_protocol.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))

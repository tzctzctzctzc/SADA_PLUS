"""Freeze a new ImageCoDe pilot before inspecting predictions; preserve temporal order."""
import json,hashlib,collections,zipfile
from pathlib import Path
ROOT=Path('/vepfs/vlms/1');assert Path.cwd().resolve()==ROOT
seed='P023-imagecode-v1'
def stable(x):return hashlib.sha256((seed+':'+x).encode()).hexdigest()
def parent(k):return k.split('-shot')[0]
def kind(k):return 'static' if k.startswith('open-images-') else 'video'
src=ROOT/'data/ImageCoDe/github/data'
train=json.loads((src/'train_data.json').read_text());valid=json.loads((src/'valid_data.json').read_text())
assert not set(map(parent,train)) & set(map(parent,valid))
def row(directory,gold,caption,split):
 return {'id':directory+':'+gold,'directory':directory,'cluster':parent(directory),'kind':kind(directory),'gold':int(gold),'caption':caption,'split':split}
cal=[];used=set()
for gold in map(str,range(10)):
 options=[row(k,gold,v[gold],'calibration') for k,v in train.items() if gold in v]
 for r in sorted(options,key=lambda r:stable(r['id'])):
  if r['cluster'] in used:continue
  cal.append(r);used.add(r['cluster'])
  if sum(x['gold']==int(gold) for x in cal)==4:break
assert len(cal)==40
groups=collections.defaultdict(list)
for k,v in valid.items():
 for gold,caption in v.items():groups[parent(k)].append(row(k,gold,caption,'evaluation'))
selected=sorted(groups,key=stable)[:256]
evaluation=[min(groups[k],key=lambda r:stable(r['id'])) for k in selected]
cfg={'id':'P023','purpose':'New-data test of patch readout against corrected VLM outputs and SigLIP2; no novelty claim.',
 'seed':seed,'model':'Qwen2.5-VL-7B-Instruct','retrieval_model':'siglip2-so400m-patch14-384',
 'layer_range':[13,27],'topk':2,'sharpen_power':5,
 'min_pixels':4*28*28,'max_pixels':448*448,'n_images':10,'labels':list('ABCDEFGHIJ'),
 'repetition_penalty':1.0,'backend':'sdpa','image_order':'original chronological order; no permutation',
 'calibration':cal,'evaluation':evaluation,
 'calibration_policy':'Four training questions per target label, distinct source-video/static-set clusters; no train/valid source-video overlap.',
 'evaluation_policy':'One hash-selected caption per hash-selected validation source-video/static-set group; 256 distinct groups.',
 'remaining_validation_clusters':[x for x in sorted(groups) if x not in set(selected)],
 'source_hashes':{str(src/n):hashlib.sha256((src/n).read_bytes()).hexdigest() for n in ['train_data.json','valid_data.json']},
 'metrics':['accuracy','paired cluster bootstrap 95CI','static/video breakdown','MRR','image-length audit'],
 'methods':['raw_logits','all_span_attention_calibrated','patch_attention_calibrated','patch_density_calibrated','patch_on_total_layers','code_AGD_raw','SigLIP2'],
 'limits':['Exploratory pilot subset, not full public benchmark or SOTA.',
  'Original chronological context retained because descriptions can be relational or temporal.',
  'Official training split is used for calibration; attention-only priors do not use labels, code AGD does.',
  'Pointwise SigLIP2 cannot condition on other images; results test this limitation without attributing all differences to reasoning.']}
p=ROOT/'configs/P023_imagecode.json'
if p.exists():assert json.loads(p.read_text())==cfg
else:p.write_text(json.dumps(cfg,indent=2))
print(json.dumps({'calibration':len(cal),'evaluation':len(evaluation),'kinds':dict(collections.Counter(r['kind'] for r in evaluation)),'heldout_clusters':len(cfg['remaining_validation_clusters'])}))

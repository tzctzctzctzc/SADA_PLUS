"""Exact information quantities for a finite protocol-permutation design.

For K equiprobable rotations, a fixed chosen image with correct answer coding
gives Y=(Z-R-C) mod K. R and C are independent; Y is independent of either
alone, but deterministic from the pair. Every nonnegative two-source PID
therefore has synergy log2(K). Decoding back to Z makes the response constant.

This is answer-code cooperation, not new visual evidence. The result does not
invalidate PID, or replicate any paper's neural BATCH estimator.
"""
import json,math,hashlib
from collections import Counter
from pathlib import Path
def entropy(values):
 count=Counter(values);n=sum(count.values())
 return -sum((v/n)*math.log2(v/n) for v in count.values())
def mutual(a,b):
 return entropy(a)+entropy(b)-entropy(list(zip(a,b)))
def quantities(rows,target):
 a=[tuple(r['image_order']) for r in rows]
 b=[tuple(r['option_order']) for r in rows]
 y=[target(r) for r in rows]
 ia,ib,ij=mutual(y,a),mutual(y,b),mutual(y,list(zip(a,b)))
 return {'H_Y_bits':entropy(y),'I_Y_image_order_bits':ia,'I_Y_option_order_bits':ib,
  'I_Y_joint_order_bits':ij,'net_synergy_bits':ij-ia-ib,
  'nonnegative_PID_synergy_bounds_bits':[max(0.,ij-ia-ib),max(0.,ij-max(ia,ib))]}
def letter(r):
 p=r['parsed_prediction']
 return ord(p)-65 if p in ['A','B','C','D'] else -1
def canonical(r):
 p=letter(r)
 return r['image_order'][r['option_order'][p]] if p>=0 else -1
toy=[]
for ri in range(4):
 io=[(i+ri)%4 for i in range(4)]
 for rc in range(4):
  co=[(i+rc)%4 for i in range(4)]
  pos=io.index(0);answer=co.index(pos)
  toy.append({'image_order':io,'option_order':co,'parsed_prediction':chr(65+answer)})
report={'id':'P012','scope':'Post-hoc diagnostic on frozen P006; exact finite 4x4 protocol distribution, not neural PID estimator reproduction',
 'theorem':{'code':'Y=(Z-R-C) mod K, independent uniform R,C and fixed Z',
 'encoded_target':quantities(toy,letter),'canonical_target':quantities(toy,canonical),
 'meaning':'Purely representational cooperation reaches log2(K) although image set and requested semantic entity are fixed; the same is true for a consistently wrong entity.'},
 'modes':{},'source_sha256':{},
 'limitations':['Protocol variables are image/choice permutations, not arbitrary visual/textual content.',
 'Only eight already-examined questions; no population statistical claim.',
 'For real responses, report bounds valid for nonnegative PID atoms, not a chosen redundancy functional.',
 'Canonicalizing the response is a diagnostic inverse of known permutations; it is not a deployable answer correction and uses no gold answer.',
 'No evidence yet that existing neural PID benchmark conclusions reverse under this control.']}
assert abs(report['theorem']['encoded_target']['nonnegative_PID_synergy_bounds_bits'][0]-2)<1e-12
assert report['theorem']['canonical_target']['H_Y_bits']==0
for mode in ['baseline','scale','canonical_start']:
 p=Path('runs/P006_permutation_'+mode+'/predictions.jsonl')
 report['source_sha256'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 rows=list(map(json.loads,p.read_text().splitlines()))
 group={}
 for q in sorted({r['base_id'] for r in rows}):
  rr=[r for r in rows if r['base_id']==q]
  assert len(rr)==16 and len({(tuple(r['image_order']),tuple(r['option_order'])) for r in rr})==16
  assert all(canonical(r)==r.get('predicted_original_image',-1) for r in rr)
  group[q]={'n':16,'correct':sum(r['correct'] for r in rr),'invalid':sum(letter(r)<0 for r in rr),
   'encoded_target':quantities(rr,letter),'canonical_target':quantities(rr,canonical),
   'canonical_predictions':dict(Counter(canonical(r) for r in rr))}
 report['modes'][mode]=group
Path('results/P012_protocol_information.json').write_text(json.dumps(report,indent=2))
print(json.dumps({'theorem':report['theorem'],'baseline':report['modes']['baseline']},indent=2))

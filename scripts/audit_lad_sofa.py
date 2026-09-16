"""Audit the pinned LAD SoFA mask before treating it as a strict inter-image baseline."""
import ast,hashlib,json,math
from pathlib import Path
import torch
p=Path('third_party/LAD/src/methods/sofa.py')
node=next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_apply_sofa_mask')
ns={'torch':torch,'math':math}
exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),ns)
n=12
mask=torch.full((n,n),torch.finfo(torch.float32).min).triu(1)[None,None]
image=torch.zeros((1,n),dtype=torch.bool);image[:,2:5]=True;image[:,7:10]=True
out=ns['_apply_sofa_mask'](mask,image,.5)
opened=(out[0,0]>-1e10)&(mask[0,0]<-1e10)
within=sum(int(opened[a:b,a:b].sum()) for a,b in [(2,5),(7,10)])
between=int(opened[2:5,7:10].sum())
text_changed=int(((out!=mask)&~(image[:,None,:,None]&image[:,None,None,:])).sum())
report={'source':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
 'pinned_repository':'brightXian/LAD','commit':'c704b9d93eb207652c0b60a5e2e83bcdcae01508',
 'toy_image_spans':[[2,5],[7,10]],'sigma':.5,
 'opened_future_edges_inside_images':within,'opened_future_edges_between_images':between,
 'changed_text_edges':text_changed,'future_edge_log_weight':float(out[0,0,2,3]),
 'interpretation':'The LAD implementation softens future attention within each image span as well as between images. Keep this variant distinct from a strict inter-image-only implementation.',
 'backend':'Pinned LAD Qwen loader uses eager. Its wrapper requires a non-None attention mask; a transfer to SDPA must materialize the causal mask.',
 'official_repository':'https://github.com/xytian1008/sofa',
 'official_repository_observation':'Only Coming soon README visible on 2026-09-17; no official implementation reproduction claimed.',
 'passed':within==6 and between==9 and text_changed==0}
assert report['passed']
Path('results/LAD_sofa_mask_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))

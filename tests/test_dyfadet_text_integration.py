from copy import deepcopy
import torch
import pytest
from test_dyfadet_uda import config,fixed_pseudo
from tools import dyfadet_text_engine as e

def textconfig(tmp_path,variant):
 cfg=config(tmp_path);p=tmp_path/'text.pt'
 labels=['TAKE','PUT','WASH','OPEN','CLOSE','INSERT','TURN_ON','CUT','TURN_OFF','POUR']
 torch.save(dict(embeddings=torch.randn(10,4,24),labels=labels),p)
 cfg['benchmark']['text']=dict(cache=str(p),sha256=e.sha(p),variant=variant)
 return cfg

@pytest.mark.parametrize('variant',['tifad','pda'])
def test_full_model_optimizer_ema_resume_routing_and_text(variant,tmp_path,monkeypatch):
 cfg=textconfig(tmp_path,variant);ctx=e.build(cfg,'uda','cpu',0);m=ctx['model'];m.eval()
 val=next(iter(ctx['val']))
 with torch.no_grad():
  orig=e._ORIGINAL_DENSE(m.detector,val);new=e.base.dense(m,val)
  assert e.base.digest(orig)==e.base.digest(new)
  m.enabled=False;assert e.base.digest(e.base.dense(m,val))==e.base.digest(orig);m.enabled=True
 m.loss_normalizer=17.;assert m.detector.loss_normalizer==17.
 assert e.base.digest(m.detector.state_dict())==ctx['initial_sha']
 assert e.base.digest(m.state_dict())==e.base.digest(ctx['ema'].module.state_dict())
 src=next(iter(ctx['train']));target=next(iter(ctx['target']));monkeypatch.setattr(e,'pseudo',fixed_pseudo)
 for _ in range(4):loss,norms=e.update(ctx,src,target,True)
 assert all(v>0 for v in norms.values()) and set(norms)==set(e.MODULES)
 assert all(p.grad is None for p in ctx['ema'].module.parameters())
 assert sum(float(p.grad.abs().sum()) for p in m.semantic.text_projection.parameters() if p.grad is not None)>0
 saved=deepcopy(e.capture(ctx,1,[],[]));e.restore(ctx,deepcopy(saved))
 assert e.base.digest(e.capture(ctx,1,[],[]))==e.base.digest(saved)
 l1,_=e.update(ctx,src,target);after=e.base.digest(e.capture(ctx,1,[],[]))
 e.restore(ctx,deepcopy(saved));l2,_=e.update(ctx,src,target)
 assert l1==l2 and after==e.base.digest(e.capture(ctx,1,[],[]))
 m.eval();un=[{k:v for k,v in x.items() if k not in ('segments','labels')} for x in val]
 with torch.no_grad():assert e.base.digest(m(val))==e.base.digest(e.predict_unlabeled(m,un))
 m.last_semantic=None

def test_text_cache_tamper_rejected(tmp_path):
 cfg=textconfig(tmp_path,'pda');cfg['benchmark']['text']['sha256']='bad'
 with pytest.raises(AssertionError):e.build(cfg,'uda','cpu',0)

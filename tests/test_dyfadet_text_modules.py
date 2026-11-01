from copy import deepcopy
import pytest
import torch
from tools.dyfadet_text_modules import SemanticPyramid,masked_softmax,salient_mask
from vendor.tifad.tv_attn import TVCA

@pytest.mark.parametrize('variant',['tifad','pda'])
def test_full_strength_padding_text_and_visual_gradients(variant):
 torch.set_num_threads(2);torch.manual_seed(11)
 net=SemanticPyramid(torch.randn(10,4,24),variant,dim=32,levels=2)
 features=[torch.randn(2,32,16,requires_grad=True),torch.randn(2,32,8,requires_grad=True)]
 masks=[torch.ones(2,1,x.shape[-1],dtype=torch.bool) for x in features]
 for m in masks:m[1,:,-3:]=False
 features=[x*m for x,m in zip(features,masks)]
 cf,rf,logits,*_=net(features,masks)
 assert any(not torch.equal(a,b) for a,b in zip(cf,features))
 assert any(not torch.equal(a,b) for a,b in zip(rf,features))
 assert any(torch.count_nonzero(x)>0 for x in logits)
 opt=torch.optim.AdamW(net.parameters(),lr=.001)
 for step in range(4):
  fresh=[x.detach().requires_grad_() for x in features]
  c,r,l,h,s,obs=net(fresh,masks)
  loss=sum((a-.4).square().mean()+b.square().mean()+(v-.2).square().mean()+f.square().mean() for a,b,v,f in zip(c,r,l,h))
  opt.zero_grad();loss.backward()
  assert all(x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum()>0 for x in fresh)
  opt.step()
 assert sum(float(p.grad.abs().sum()) for p in net.text_projection.parameters() if p.grad is not None)>0
 with torch.no_grad():
  a=net(features,masks);saved=net.text_embeddings.clone();net.text_embeddings.zero_();b=net(features,masks);net.text_embeddings.copy_(saved)
 assert any(not torch.equal(x,y) for x,y in zip(a[1],b[1])),'text must affect regression features'
 assert any(not torch.equal(x,y) for x,y in zip(a[2],b[2]))
 assert all(torch.count_nonzero(x.transpose(1,2)[~m.squeeze(1)])==0 for x,m in zip(a[0],masks))
 other=deepcopy(net);other.load_state_dict(net.state_dict(),strict=True)
 assert all(torch.equal(x,y) for x,y in zip(net(features,masks)[2],other(features,masks)[2]))

def test_empty_mask_and_salience_ignore_padding():
 torch.set_num_threads(2)
 x=torch.randn(2,7,8);text=torch.randn(2,10,8);valid=torch.ones(2,7,dtype=torch.bool);valid[0]=False;valid[1,4:]=False
 weights=masked_softmax(torch.randn(2,7),valid,-1)
 assert not weights[0].any() and torch.allclose(weights[1].sum(),torch.tensor(1.))
 a=salient_mask(x,text,valid,torch.zeros(2,7));x[~valid]=10000
 b=salient_mask(x,text,valid,torch.zeros(2,7))
 assert torch.equal(a,b) and not a[~valid].any() and torch.isfinite(a).all()

def test_author_attention_masks_follow_each_sample_not_other_batch_member():
 torch.set_num_threads(2);torch.manual_seed(19)
 m=TVCA(16,16,16,4,dropout=0.).eval()
 v=torch.randn(2,8,16);t=torch.randn(2,10,16);sam=torch.rand(2,8,1)
 valid=torch.ones(2,8,dtype=torch.bool);valid[1,-3:]=False
 with torch.no_grad():
  both=m(v,t,sam,attention_mask_v=valid)
  for i in range(2):
   single=m(v[i:i+1],t[i:i+1],sam[i:i+1],attention_mask_v=valid[i:i+1])
   for a,b in zip(both,single):torch.testing.assert_close(a[i:i+1],b,atol=2e-6,rtol=1e-5)
  changed=v.clone();changed[~valid]=10000
  masked=m(changed,t,sam,attention_mask_v=valid)
  torch.testing.assert_close(masked[1],both[1],atol=2e-6,rtol=1e-5)

import itertools
from types import SimpleNamespace
import torch
from tools.phase_boundary import (path_partition,phase_posterior,map_coordinates,strong_view,
 sample,PhaseBoundary,sequence_loss,source_loss,target_loss)

def test_ordered_partition_matches_all_variable_dwell_paths():
 x=torch.randn(2,7,3,dtype=torch.float64,requires_grad=True);lp=x.log_softmax(-1)
 paths=[]
 for a in range(1,6):
  for b in range(a+1,7):
   paths.append(lp[:,:a,0].sum(-1)+lp[:,a:b,1].sum(-1)+lp[:,b:,2].sum(-1))
 assert torch.allclose(path_partition(lp),torch.stack(paths).logsumexp(0),atol=1e-10)
 post=phase_posterior(x);assert torch.allclose(post.sum(-1),torch.ones(2,7,dtype=x.dtype),atol=1e-9)
 assert torch.allclose(post[:,0,0],torch.ones(2,dtype=x.dtype)) and torch.allclose(post[:,-1,2],torch.ones(2,dtype=x.dtype))
 path_partition(lp).sum().backward();assert x.grad.isfinite().all() and x.grad.abs().sum()>0

def test_order_sensitive_and_not_equal_thirds():
 x=torch.full((1,9,3),-10.);x[:,:1,0]=10;x[:,1:7,1]=10;x[:,7:,2]=10
 assert path_partition(x.log_softmax(-1))>path_partition(x.flip(-1).log_softmax(-1))
 post=phase_posterior(x);assert post.argmax(-1).tolist()==[[0,1,1,1,1,1,1,2,2]]

def test_crop_rescale_exact_physical_mapping_and_target_gt_absent():
 v={'feats':torch.arange(100.).repeat(4,1),'segments':torch.tensor([[1.,2.]]),'labels':torch.tensor([2])}
 y,view=strong_view(v,2,0);assert not y['segments'].numel() and not y['labels'].numel()
 coords=torch.tensor([view['lo'],view['lo']+4.75,view['lo']+view['width']-1.])
 assert torch.allclose(map_coordinates(map_coordinates(coords,view),view,True),coords,atol=1e-5)
 mapped=map_coordinates(torch.arange(view['lo'],view['lo']+view['width']).float(),view)
 assert torch.isfinite(mapped).all()

def test_linear_sampling_exact_centers_and_endpoint_feature_derivatives():
 x=torch.arange(12.).repeat(3,1).requires_grad_();q=torch.tensor([1.,3.5,8.],requires_grad=True)
 y=sample(x,q,1);assert torch.allclose(y[:,0],q,atol=1e-6)
 y.sum().backward();assert x.grad.abs().sum()>0 and torch.allclose(q.grad,torch.full_like(q,3.),atol=1e-5)

def test_actual_text_phase_network_has_text_and_temporal_gradients():
 torch.manual_seed(123);h=PhaseBoundary(torch.randn(10,4,512));x=torch.randn(3,24,512,requires_grad=True)
 loss,_,_=sequence_loss(h,x,torch.tensor([0,1,2]));loss.backward()
 assert torch.isfinite(loss) and x.grad.norm()>0 and h.text_proj.weight.grad.norm()>0
 assert h.cross.in_proj_weight.grad.norm()>0
 clone=PhaseBoundary(torch.randn(10,4,512));clone.load_state_dict(h.state_dict(),strict=True)
 assert torch.equal(h.text,clone.text)

def test_source_gt_trains_both_offsets_and_independent_error_scales():
 torch.manual_seed(123);h=PhaseBoundary(torch.randn(10,4,512));f=[torch.randn(1,512,32,requires_grad=True)]
 mask=[torch.ones(1,1,32,dtype=torch.bool)];reg=[torch.full((1,2,32),4.,requires_grad=True)]
 v=[{'feats':torch.zeros(512,32),'segments':torch.tensor([[4.,13.],[18.,25.]]),'labels':torch.tensor([0,1])}]
 loss,stats=source_loss(SimpleNamespace(fpn_strides=[1]),h,f,mask,reg,v);loss.backward()
 assert stats['source_instances']==2 and torch.isfinite(loss)
 assert f[0].grad.norm()>0 and reg[0].grad.norm()>0 and h.scale.weight.grad.norm()>0
 assert reg[0].grad[:,0].norm()>0 and reg[0].grad[:,1].norm()>0

def test_target_empty_support_is_finite_zero_and_does_not_invent_labels():
 h=PhaseBoundary(torch.randn(10,4,512));f=[torch.randn(1,512,16,requires_grad=True)];m=[torch.ones(1,1,16,dtype=torch.bool)]
 c=[torch.full((1,97,16),-20.)];r=[torch.ones(1,2,16)]
 model=SimpleNamespace(fpn_strides=[1],phase_boundary=h)
 loss,stats=target_loss(model,model,h,(f,m,c,r),(f,m,c,r),[dict(original_length=16,lo=0,width=16,ratio=1,length=16)],1)
 assert loss==0 and stats['target_proposals']==0;loss.backward();assert f[0].grad.abs().sum()==0

def test_target_endpoint_gate_separates_left_and_right_and_trains_real_regression():
 class Head(torch.nn.Module):
  def __init__(self,teacher):super().__init__();self.teacher=teacher
  def sequence(self,x):
   logits=x.mean(-1)[:,None,:,None].expand(-1,10,-1,3)
   scores=x.mean((-1,-2))[:,None].expand(-1,10).clone();scores[:,0]=scores[:,0]+10
   return logits,scores
  def dense(self,x):
   # 教师左端点更可靠，右端点更不可靠；必须只训练左边。
   scales=x.new_tensor([.1,2.] if self.teacher else [1.,1.]).expand(len(x),-1)
   return x.new_full((len(x),2),6.),scales,x.new_ones(len(x))
  def endpoint_error(self,x,reference,coordinates,label,side,stride,length,mask=None):
   return coordinates.new_full(coordinates.shape,([.1,2.] if self.teacher else [1.,1.])[side])
 student=Head(False);teacher=SimpleNamespace(fpn_strides=[1],phase_boundary=Head(True))
 f=[torch.randn(1,512,32,requires_grad=True)];m=[torch.ones(1,1,32,dtype=torch.bool)]
 cls=torch.full((1,97,32),-20.);cls[0,0,16]=10
 r=[torch.full((1,2,32),4.,requires_grad=True)]
 teacher_r=[torch.full((1,2,32),3.)]
 tl,stats=target_loss(SimpleNamespace(fpn_strides=[1]),teacher,student,(f,m,[cls],teacher_r),(f,m,[cls.clone().requires_grad_()],r),[dict(original_length=32,lo=0,width=32,ratio=1,length=32)],1)
 assert stats['target_left_used']==1 and stats['target_right_used']==0
 tl.backward();assert r[0].grad[0,0,16]!=0 and r[0].grad[0,1,16]==0
 assert f[0].grad.norm()>0

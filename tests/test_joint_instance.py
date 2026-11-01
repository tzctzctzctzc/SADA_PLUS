import torch
import pytest
from tools.joint_instance import JointHead,sample,pair_iou,diou,source_objective,source_proposals,select,base_at_boxes
torch.set_num_threads(1)

def packet():
    return dict(fpn=[torch.randn(1,8,64,requires_grad=True),torch.randn(1,8,32,requires_grad=True)],
        masks=[torch.ones(1,64,dtype=torch.bool),torch.ones(1,32,dtype=torch.bool)],
        points=[torch.stack((torch.arange(64),torch.zeros(64),torch.ones(64)*1e5,torch.ones(64)),1),torch.stack((torch.arange(32)*2,torch.zeros(32),torch.ones(32)*1e5,torch.ones(32)*2),1)],
        logits=[torch.randn(1,64,10,requires_grad=True),torch.randn(1,32,10,requires_grad=True)],offsets=[torch.ones(1,64,2)*4,torch.ones(1,32,2)*4])

def test_linear_interpolation_and_padding():
    f=torch.arange(6.).reshape(1,1,6).requires_grad_();m=torch.tensor([[1,1,1,1,0,0]],dtype=torch.bool)
    v,s=sample(f,m,torch.tensor([[.5,2.,4.,-2.]]),1)
    assert torch.allclose(v[0,:,0],torch.tensor([.5,2.,0.,0.])) and s.tolist()==[[True,True,False,False]]
    v.sum().backward();assert f.grad[0,0,4:].count_nonzero()==0

def test_zero_boundary_initialization_and_two_stage_identity():
    p=packet();h=JointHead(8,2);b=torch.tensor([[12.,24.],[25.,38.]])
    out=h(p,b,base_at_boxes(p,b));assert len(out)==2
    for x in out:assert torch.equal(x['boxes'],b)

def test_source_full_gt_jitter_not_fixed_candidate_budget():
    p=packet();gt=torch.tensor([[10.,18.],[30.,40.]])
    b,l=source_proposals(p,dict(feats=torch.zeros(8,64),segments=gt),0)
    assert len(b)>=len(gt)*5 and any(torch.equal(x,gt[0]) for x in b)
    assert l.shape==(len(b),10)

def test_source_quality_uses_class_and_iou_not_any_class_overlap():
    b=torch.tensor([[10.,20.],[10.,20.]],requires_grad=True);l=torch.zeros(2,10,requires_grad=True);q=torch.zeros(2,10,requires_grad=True)
    out=[dict(boxes=b,logits=l,quality=q,supported=torch.ones(2,dtype=torch.bool))]*2
    loss,_=source_objective(out,torch.tensor([[10.,20.]]),torch.tensor([3]));loss.backward()
    assert q.grad[:,3].lt(0).all() and q.grad[:,2].gt(0).all()

def test_joint_gradient_into_both_scales_encoder_and_base_classifier():
    p=packet();h=JointHead(8,2)
    for s in h.stages:
        torch.nn.init.normal_(s.cls.weight,std=.01);torch.nn.init.normal_(s.boundary.weight,std=.01);torch.nn.init.normal_(s.quality.weight,std=.01)
    b=torch.tensor([[10.,22.],[30.,42.]])
    out=h(p,b,base_at_boxes(p,b));loss,_=source_objective(out,torch.tensor([[11.,23.],[31.,43.]]),torch.tensor([2,4]));loss.backward()
    assert all(x.grad is not None and x.grad.norm()>0 for x in p['fpn'])
    assert all(s.encoder.layers[0].self_attn.in_proj_weight.grad.norm()>0 for s in h.stages)
    assert p['logits'][0].grad.norm()>0

def test_stage_two_really_resamples_changed_boundaries():
    p=packet();h=JointHead(8,2);b=torch.tensor([[12.,24.]])
    with torch.no_grad():h.stages[0].boundary.bias[0]=1
    seen=[]
    handle=h.stages[1].register_forward_pre_hook(lambda m,a:seen.append(a[1].detach().clone()))
    out=h(p,b,base_at_boxes(p,b));handle.remove();assert torch.equal(seen[0],out[0]['boxes'].detach()) and not torch.equal(seen[0],b)

def test_boundary_decode_always_positive_and_diou_gradient():
    a=torch.tensor([[2.,9.]],requires_grad=True);b=torch.tensor([[3.,10.]])
    x=diou(a,b);assert x.item()>0;x.sum().backward();assert a.grad.norm()>0
    assert pair_iou(b,b).item()==1

def test_source_selection_epoch_zero_ties_and_first_stop():
    h=[dict(epoch=e,map=.5) for e in range(0,11,2)]
    assert select(h)==(h[0],True)
    assert not select(h[:-1])[1]

def test_missing_source_history_rejected():
    with pytest.raises(AssertionError):select([dict(epoch=0,map=.5),dict(epoch=4,map=.6)])

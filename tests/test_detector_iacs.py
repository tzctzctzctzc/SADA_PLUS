import torch


def test_temporal_quality_targets_are_class_specific_and_detached():
    from tools.detector_iacs_math import iou_targets
    classes=torch.tensor([[0.,1.,0.],[1.,0.,1.],[0.,0.,0.]])
    gt=torch.tensor([[2.,4.],[1.,1.],[0.,0.]]);pred=torch.tensor([[3.,2.],[1.,1.],[2.,2.]],requires_grad=True)
    q=iou_targets(classes,gt,pred);assert not q.requires_grad
    assert torch.allclose(q,torch.tensor([[0.,4/7,0.],[1.,0.,1.],[0.,0.,0.]]))


def test_varifocal_author_formula_value_and_grad_update():
    from tools.detector_iacs_math import original_varifocal
    torch.manual_seed(9);p=torch.randn(8,3,requires_grad=True);r=p.detach().clone().requires_grad_();q=torch.tensor([[0.,.7,0.]]*8)
    a=original_varifocal()(p,q,reduction='sum');weight=torch.where(q>0,q,.75*r.sigmoid().square());b=(torch.nn.functional.binary_cross_entropy_with_logits(r,q,reduction='none')*weight).sum();assert torch.equal(a,b);a.backward();b.backward();assert torch.equal(p.grad,r.grad)
    x=torch.optim.AdamW([p],lr=1e-5);y=torch.optim.AdamW([r],lr=1e-5);x.step();y.step();assert torch.equal(p,r)


def test_ema_update_keeps_teacher_gradient_free_and_full_state():
    from tools.detector_iacs_math import ema_step
    a=torch.nn.Linear(3,2);b=torch.nn.Linear(3,2);a.requires_grad_(False);old={k:v.clone() for k,v in a.state_dict().items()};ema_step(a,b,.75)
    for k,v in a.state_dict().items():assert torch.equal(v,old[k]*.75+b.state_dict()[k]*.25)


def test_ap_quality_uses_fixed_five_thresholds_without_overpenalizing_valid_boxes():
    from tools.detector_iacs_math import iou_targets
    c=torch.ones(3,1);gt=torch.tensor([[1.,1.]]*3);pred=torch.tensor([[1.,1.],[3.,3.],[10.,10.]])
    q=iou_targets(c,gt,pred,ap_aligned=True)
    assert torch.equal(q,torch.tensor([[1.],[.6],[.2]]))

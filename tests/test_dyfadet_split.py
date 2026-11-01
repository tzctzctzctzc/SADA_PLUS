from copy import deepcopy
import torch
import pytest
from test_dyfadet_uda import config,fixed_pseudo
from tools import dyfadet_uda_engine as e
from tools.dyfadet_split_policy import temporal_view,inverse_segments,stability_filter,assignment_metadata


def prediction(boxes,labels=None):
    a=torch.tensor(boxes,dtype=torch.float32).reshape(-1,2)
    return dict(segments=a,labels=torch.tensor(labels or [0]*len(a)),scores=torch.ones(len(a)))


def test_pixel_center_mapping_actual_rounded_ratio():
    v=dict(feats=torch.arange(17.).reshape(1,17),crop_length=17)
    changed,ratio=temporal_view(v,.9)
    assert changed['crop_length']==15 and ratio==15/17
    x=torch.tensor([[2.,9.]])
    assert torch.allclose(inverse_segments((x+.5)*ratio-.5,ratio,17),x)
    # Linear ramp interpolation obeys the same pixel-center mapping away from padding.
    positions=(torch.arange(15.)+.5)/ratio-.5
    assert torch.allclose(changed['feats'][0,1:-1],positions[1:-1],atol=1e-6)
    with pytest.raises(AssertionError):temporal_view(dict(v,segments=x),.9)


def test_stability_requires_both_views_class_and_endpoint():
    p=prediction([[0,10],[20,30],[40,50],[60,70]])
    v1=prediction([[.5,10.5],[20,30],[40,50],[62,72]],[0,1,0,0])
    v2=prediction([[0,10],[20,30],[45,55],[60,70]])
    keep,_,_=stability_filter(p,[v1,v2])
    assert keep.tolist()==[True,False,False,False]
    assert not stability_filter(p,[v1,prediction([])])[0].any()


def test_mutual_match_rejects_duplicate_support():
    p=prediction([[0,10],[.1,10.1]])
    v=prediction([[0,10]])
    assert stability_filter(p,[v,v])[0].tolist()==[True,False]
    assert len(stability_filter(prediction([]),[v,v])[0])==0


def test_confidence_assignment_matches_official_overlap_and_ties(tmp_path):
    ctx=e.build(config(tmp_path),'uda','cpu',0);m=ctx['model']
    videos=next(iter(ctx['target']));points=e.base.dense(m,videos)[0]
    for seg,lab in [(torch.tensor([[4.,12.],[4.,12.],[7.,10.],[1.,40.]]),torch.tensor([1,2,3,1])),
                    (torch.zeros(0,2),torch.zeros(0,dtype=torch.long))]:
        score=torch.linspace(.5,.9,len(seg))
        target,offset,weight=assignment_metadata(m,points,seg,lab,score)
        a,b=m.label_points(points,[seg],[lab])
        assert torch.equal(target,a[0]) and torch.equal(offset,b[0])
        assert bool(((weight>=0)&(weight<=1)).all())
    seg=torch.tensor([[4.,12.],[4.,12.]])
    _,_,weight=assignment_metadata(m,points,seg,torch.tensor([1,2]),torch.tensor([.9,.6]))
    assert torch.all(weight[weight>0]==.6)


def test_confidence_weights_cls_only_and_unstable_offsets_get_zero_gradient(tmp_path):
    ctx=e.build(config(tmp_path),'uda','cpu',0);m=ctx['model'];m.eval()
    videos=next(iter(ctx['target']))[:1];out=e.base.dense(m,videos);pred=fixed_pseudo(m,videos)
    c,r,n,nr,w=e.positive_loss(m,out,pred)
    changed=deepcopy(pred)
    for p in changed:p['scores']*=.5
    c2,r2,*_=e.positive_loss(m,out,changed)
    assert torch.allclose(c2,c*.5) and torch.equal(r,r2)
    for p in pred:p['regression_keep'].zero_()
    c,r,n,nr,w=e.positive_loss(m,out,pred)
    assert n>0 and nr==0 and r.item()==0
    grads=torch.autograd.grad(c+r,[p for p in m.reg_head.parameters() if p.requires_grad],allow_unused=True)
    assert all(x is None or not x.any() for x in grads)
    assert c.item()>0


def test_regression_separate_assignment_can_ignore_short_unstable_instance(tmp_path):
    ctx=e.build(config(tmp_path),'uda','cpu',0);m=ctx['model'];m.eval()
    videos=next(iter(ctx['target']))[:1];out=e.base.dense(m,videos)
    pred=[dict(segments=torch.tensor([[4.,12.],[7.,9.]]),labels=torch.tensor([2,2]),
               scores=torch.tensor([.9,.8]),regression_keep=torch.tensor([True,False])) for _ in videos]
    c,r,n,nr,w=e.positive_loss(m,out,pred)
    only=[{k:v[:1] for k,v in p.items()} for p in pred]
    _,rr,_,nnr,_=e.positive_loss(m,out,only)
    assert torch.equal(r,rr) and nr==nnr and nr>0

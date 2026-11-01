import torch
from tools.regression_gram import gram_alignment,augmented_svd,inverse_from_spectrum,regression_inputs


def test_compact_inverse_matches_full_gram_spectrum():
    torch.manual_seed(123);x=torch.randn(20,7,dtype=torch.float64)
    e,v=augmented_svd(x);a=torch.cat([torch.ones(20,1,dtype=x.dtype),x],1)
    vals,basis=torch.linalg.eigh(a.T@a);order=torch.argsort(vals,descending=True)
    rank=5;reference=(basis[:,order[:rank]]/vals[order[:rank]])@basis[:,order[:rank]].T
    assert torch.allclose(inverse_from_spectrum(e,v,rank),reference,rtol=1e-10,atol=1e-10)


def test_finite_real_gradient_to_both_domains_and_no_input_mutation():
    torch.manual_seed(123);a=torch.randn(24,7,dtype=torch.float64,requires_grad=True)
    b=(torch.randn(24,7,dtype=torch.float64)*2+.3).requires_grad_();aa=a.detach().clone();bb=b.detach().clone()
    loss,info=gram_alignment(a,b);loss.backward()
    assert loss>0 and info['rank']>0
    assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all() and a.grad.norm()>0 and b.grad.norm()>0
    assert torch.equal(a,aa) and torch.equal(b,bb)


def test_same_features_zero_angle_and_scale_with_well_supported_columns():
    torch.manual_seed(11);a=torch.randn(30,6,dtype=torch.float64)
    loss,info=gram_alignment(a,a.clone())
    assert loss<1e-10 and info['scale']==0


def test_exact_original_three_tap_convolution_layout():
    from torch import nn
    class Head:
        head=[];norm=[]
    f=torch.arange(20.).reshape(1,2,10);m=torch.ones(1,1,10,dtype=torch.bool)
    inputs=regression_inputs(Head(),[f],[m]);conv=nn.Conv1d(2,2,3,padding=1)
    with torch.no_grad():
        conv.weight.copy_(torch.arange(12.).reshape_as(conv.weight)-6)
        conv.bias.copy_(torch.tensor([1.,-2.]))
    expected=conv(f).transpose(1,2)
    actual=torch.nn.functional.linear(inputs,conv.weight.flatten(1),conv.bias)
    assert torch.equal(actual,expected)


def test_too_few_points_rejected():
    import pytest
    with pytest.raises(ValueError):gram_alignment(torch.ones(3,6),torch.ones(4,6))


def test_feature_path_keeps_backbone_gradient_without_detaching_cache_helper():
    from torch import nn
    from tools.regression_gram import differentiable_detector_features
    class Backbone(nn.Module):
        def __init__(self):
            super().__init__();self.weight=nn.Parameter(torch.tensor(2.))
        def forward(self,x,m):return [x*self.weight],m,None
    class Neck(nn.Module):
        def forward(self,f,m):return f,m
    class Model:
        backbone_source=Backbone();neck_source=Neck();dropout=nn.Identity()
        backbone_target=backbone_source;neck_target=neck_source
        def preprocessing(self,video,mode):return video[0],None
    model=Model();f,m=differentiable_detector_features(model,torch.ones(1,2,3))
    assert f[0].requires_grad and torch.equal(f[0],torch.ones_like(f[0])*2)
    f[0].sum().backward();assert model.backbone_source.weight.grad>0

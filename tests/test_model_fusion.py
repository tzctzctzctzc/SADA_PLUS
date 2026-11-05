import torch
from tools.model_fusion import fuse_values,candidates,decoded


def test_equal_probabilities_and_score_weighted_geometry_without_nan():
    a=torch.tensor([[.2,.8],[0.,0.]]);b=torch.tensor([[.6,.4],[0.,0.]])
    oa=torch.ones(2,2,2)*2;ob=torch.ones_like(oa)*6
    p,o=fuse_values(a,b,oa,ob)
    assert torch.allclose(p,torch.tensor([[.4,.6],[0.,0.]]))
    assert torch.allclose(o[0],torch.tensor([[5.,5.],[10/3,10/3]])) and torch.isfinite(o).all()
    assert torch.equal(a,torch.tensor([[.2,.8],[0.,0.]]))


def test_filtering_keeps_point_class_correspondence_and_stride():
    p=torch.tensor([[.1,.7],[.8,.01]]);o=torch.tensor([[[1.,1.],[2.,3.]],[[4.,5.],[0.,0.]]])
    points=torch.tensor([[10.,0.,100.,2.],[20.,0.,100.,4.]])
    segments,scores,labels=candidates(p,o,points,.2,1,0.)
    assert labels.tolist()==[0] and scores.tolist()==p[1,:1].tolist()
    assert segments.tolist()==[[4.,40.]]
    empty=candidates(torch.zeros_like(p),o,points,.2,10,0.)
    assert all(x.shape[0]==0 for x in empty)


def test_trident_signed_logits_neighbour_orientation():
    class Model:
        use_trident_head=True;num_bins=2
        def decode_offset(self,o,left,right,mode):
            assert mode=='test' and left.shape==right.shape==(2,3,3)
            assert left[0].tolist()==[[0.,0.,1.],[0.,1.,2.],[1.,2.,3.]]
            assert right[0].tolist()==[[4.,5.,6.],[5.,6.,0.],[6.,0.,0.]]
            return torch.stack((left.sum(-1),right.sum(-1)),-1)
    raw=dict(offsets=[torch.zeros(1,3,6)],left=[torch.tensor([[[1.,2.,3.],[7.,8.,9.]]])],right=[torch.tensor([[[4.,5.,6.],[10.,11.,12.]]])])
    result=decoded(Model(),raw,0)
    assert result.shape==(3,2,2) and result[2,0].tolist()==[6.,6.]

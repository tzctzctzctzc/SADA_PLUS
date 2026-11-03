import torch


def test_scale_one_and_no_views_preserve_original_tensors_exactly():
    from tools.detector_scale_math import rescale_raw,fuse_heads
    x=torch.randn(21,4);a,s=rescale_raw(x,1.);assert a is x and s==1
    cls=torch.randn(10,32);reg=torch.rand(2,32);c,r=fuse_heads(cls,reg,[]);assert c is cls and r is reg


def test_physical_grid_roundtrip_and_regression_units():
    from tools.detector_scale_math import rescale_raw,align_head,fuse_heads
    raw=torch.arange(21).float()[:,None]
    for scale in (.75,1.25):
        x,ratio=rescale_raw(raw,scale);assert x[0]==0 and x[-1]==20
        aligned,valid=align_head(x.T,torch.ones(len(x),dtype=torch.bool),21,ratio)
        assert torch.allclose(aligned[0,valid],raw[:,0][valid],atol=2e-6,rtol=0)
        cls=torch.zeros(10,21);reg=torch.ones(2,21)*3;vc=torch.zeros(10,len(x));vr=torch.ones(2,len(x))*3*ratio
        c,r=fuse_heads(cls,reg,[(vc,vr,torch.ones(len(x),dtype=torch.bool),ratio)])
        assert torch.equal(c,cls) and torch.allclose(r,reg,atol=1e-6,rtol=0)


def test_invalid_scaled_neighbors_do_not_change_original_head():
    from tools.detector_scale_math import fuse_heads
    c=torch.zeros(10,11);r=torch.ones(2,11);v=torch.zeros(16,dtype=torch.bool)
    a,b=fuse_heads(c,r,[(torch.ones(10,16)*4,torch.ones(2,16)*7,v,1.25)])
    assert torch.equal(a,c) and torch.equal(b,r)


def test_runner_has_fixed_scope_and_no_training_or_gt_inference():
    from tools.run_detector_scale import SPEC,TASKS
    assert len(TASKS)==4 and SPEC['scales']==[.75,1.,1.25] and SPEC['learning_steps']==0 and SPEC['primary']=='score_boundary'

from types import SimpleNamespace
import numpy as np
import torch
from tools.trident_uncertainty import DistributionObserver,distributions,to_seconds
from tools.uncertainty_analysis import features,match_quality,auc,source_partition,fit_logistic,predict,metrics


class FakeModel:
    cfg={'model':{}};test_pre_nms_thresh=.01;test_pre_nms_topk=4;num_class_orig=2;test_duration_thresh=0
    def decode_offset(self,offsets,start,end,mode):
        left,right=distributions(offsets,start,end)
        return torch.stack((left@torch.arange(2,-1,-1,dtype=left.dtype),right@torch.arange(3,dtype=right.dtype)),-1)
    def inference_single_video(self,points,masks,classification,offsets,start,end):
        logits=classification[0];score=logits.sigmoid().flatten();score,order=score.sort(descending=True);score=score[:4];order=order[:4]
        index=order//2;category=order%2;decoded=self.decode_offset(offsets[0],start[0],end[0],'test')[category,index]
        p=points[0][index];segments=torch.stack((p[:,0]-decoded[:,0]*p[:,3],p[:,0]+decoded[:,1]*p[:,3]),-1)
        return dict(segments=segments,scores=score,labels=category)


def test_observer_records_same_candidates_without_changing_outputs():
    model=FakeModel();generator=torch.Generator().manual_seed(1)
    p=torch.tensor([[2.,0,100,1],[3.,0,100,1],[4.,0,100,1.]])
    args=([p],[torch.ones(3,dtype=torch.bool)],[torch.randn(3,2,generator=generator)],
          [torch.randn(3,6,generator=generator)],[torch.randn(2,3,3,generator=generator)],[torch.randn(2,3,3,generator=generator)])
    expected=model.inference_single_video(*args)
    with DistributionObserver(model,1) as observer:
        model.inference_single_video(*args);assert observer.record is None
        actual=model.inference_single_video(*args)
        assert all(torch.equal(expected[k],actual[k]) for k in expected)
        assert observer.record['left'].shape==(4,3)
        assert observer.call_index==2
    assert model.inference_single_video.__self__ is model


def test_coordinate_conversion_keeps_original_operation_order():
    record=dict(segments=torch.tensor([[.123456,3.987654],[10.,20.]]),centre=torch.tensor([2.,15.]),stride=torch.tensor([1.,2.]))
    meta=dict(feat_stride=16,fps=29.97,feat_num_frames=32,use_temporal_grid_feats=True,duration=100,video_length=200)
    got=to_seconds(record,meta)
    assert torch.equal(got['segments'],(record['segments']*16+16)/29.97)
    assert torch.equal(record['segments'],torch.tensor([[.123456,3.987654],[10.,20.]]))


def row(probability):
    return dict(left=np.array(probability,float),right=np.array(probability,float),stride=np.ones(2),centre=np.ones(2)*10,
        segments=np.array([[9.,11.],[9.,11.]]),scores=np.ones(2)*.5,labels=np.zeros(2,dtype=int),video_length=100.)


def test_equal_expected_boundaries_can_have_different_spread():
    r=row([[0,1,0],[.5,0,.5]]);_,u,_=features(r)
    assert u[0,0]==0 and u[1,0]>0
    assert u[0,2]==0 and u[1,2]>0
    scaled=dict(r,stride=r['stride']*3,centre=r['centre']*3,segments=r['segments']*3,video_length=300.)
    assert np.allclose(features(scaled)[1],u)


def test_same_class_matching_retains_wrong_class_and_background_errors():
    quality,error,matched=match_quality(np.array([[0.,2.],[0.,2.],[5.,6.]]),np.array([0,1,0]),np.array([[0.,2.]]),np.array([0]))
    assert quality.tolist()==[1,0,0] and matched.tolist()==[True,False,True]
    assert np.isnan(error[1]).all() and error[0].tolist()==[0,0]
    assert auc([0,1],[1,1])==.5 and auc([0,1],[0,1])==1.
    a,b=source_partition(['a','b','c','d','e']);assert not a&b and a|b==set('abcde')


def test_calibration_fit_and_prediction_are_finite_and_independent_of_global_rng():
    rng=np.random.RandomState(12);x=rng.normal(size=(240,3));y=(x[:,1]+.1*x[:,2]>0).astype(int)
    videos=np.array(['v'+str(i//20) for i in range(240)])
    state=np.random.get_state();model=fit_logistic(x[:160],y[:160],videos[:160]);p=predict(model,x[160:])
    assert all(np.array_equal(a,b) for a,b in zip(state,np.random.get_state()))
    assert metrics(y[160:],p)['auroc']>.95 and np.isfinite(p).all()

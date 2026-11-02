import numpy as np
import pandas as pd
from tools.foreground_oracle_ap import (ClassAP, RULES, calibrate, apply_rules, background_mask)
from libs.evaluation.evaluator_anet import compute_average_precision_detection


def frames(seed=5):
    rng = np.random.RandomState(seed)
    gt = pd.DataFrame({'video-id':np.repeat(['a','b'],9), 't-start':np.tile(np.arange(9),2),
                       't-end':np.tile(np.arange(9),2)+1.2, 'label':np.zeros(18,dtype=int)})
    n=140; start=rng.uniform(-1,10,n)
    pred=pd.DataFrame({'video-id':rng.choice(['a','b','empty'],n), 't-start':start,
        't-end':start+rng.uniform(.05,3,n), 'label':np.zeros(n,dtype=int),
        'score':rng.choice([.1,.2,.3,.5,.8],n), 'pbg':rng.uniform(0,1,n)})
    return gt,pred


def test_exact_official_ties_duplicates_and_filtered_sort():
    for seed in range(3):
        gt,pred=frames(seed); engine=ClassAP(gt,pred)
        for cut in [0,.2,.7,1.01]:
            mask=pred.pbg.to_numpy()<cut
            actual=engine.evaluate(mask)
            if mask.any():
                expected=compute_average_precision_detection(gt,pred[mask].reset_index(drop=True))
                np.testing.assert_allclose(actual['ap'],expected[0],atol=1e-14,rtol=0)
                np.testing.assert_array_equal(actual['tp'],expected[1])
            else:
                assert actual['map']==0 and actual['tp']==[0]*5


def test_target_calibration_tp_constraint_no_filter_fallback_and_apply():
    gt,pred=frames(); cal=calibrate(gt,pred)
    for d in cal['per_class'].values():
        selected=d['grid'][d['selected_index']]
        assert all(a>=.995*b for a,b in zip(selected['tp'],d['baseline']['tp']))
    mask=apply_rules(pred,cal,True)
    assert mask.dtype==np.bool_ and len(mask)==len(pred)
    assert (1.01,.1) in RULES


def test_background_oracle_anyclass_and_strict_threshold():
    gt,pred=frames(); pred=pred.iloc[:3].copy().reset_index(drop=True)
    pred['video-id']=['a','empty','a'];pred['t-start']=[0,0,20];pred['t-end']=[1,1,21]
    np.testing.assert_array_equal(background_mask(gt,pred),[False,True,True])


def test_zero_positive_class_cannot_tune_away_recall_constraint():
    gt,pred=frames(); pred['video-id']='empty';cal=calibrate(gt,pred)
    assert cal['per_class']['0']['selected_rule']==(1.01,.1)
    assert apply_rules(pred,cal,True).all()

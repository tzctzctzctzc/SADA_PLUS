import numpy as np
import pandas as pd
from tools.audit_background_ap_cost import IndependentAP, official, selected_indices


def test_independent_matching_matches_official_with_ties_and_replacement():
    gt = pd.DataFrame({'video-id':['a','a','b'], 't-start':[1.,3.,1.], 't-end':[2.,4.,3.]})
    pred = pd.DataFrame({'video-id':['a','a','a','a','b','c'], 't-start':[0.,1.,1.,3.,1.,0.], 't-end':[.5,2.,2.,4.,3.,1.], 'score':[.8,.6,.6,.2,.7,.9]})
    engine = IndependentAP(gt, pred)
    for mask in (np.ones(6,dtype=bool), np.array([1,0,1,1,1,1],dtype=bool), np.zeros(6,dtype=bool)):
        assert np.allclose(engine.ap(mask), official(gt, pred[mask]), atol=1e-12)


def test_deletion_budget_and_tie_do_not_touch_low_score():
    frame = pd.DataFrame({'label':[0,0,0,1,1,1], 'score':[.9,.8,.2,.9,.8,.2], 'pbg':[.5,.5,1.,.1,.9,1.]})
    assert selected_indices(frame,.5).tolist()==[0,4]
    assert selected_indices(frame,.5,True).tolist()==[1,4]

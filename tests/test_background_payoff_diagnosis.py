import numpy as np
import pandas as pd
from tools.background_payoff_diagnosis import auc,fine_rules,summarize_ap
from tools.foreground_oracle_ap import ClassAP
from libs.evaluation.evaluator_anet import compute_average_precision_detection


def test_probability_tail_resolution_and_no_filter_fallback():
    f=pd.DataFrame(dict(pbg=np.linspace(.999,.99999,101)))
    rules=fine_rules(f)
    assert any(.999<t<.99999 for t,s in rules)
    assert (1.01,.1) in rules and (.9750000000000001,.5) in rules


def test_reranked_AP_matches_official_greedy_and_ties():
    gt=pd.DataFrame({'video-id':['a','a'],'t-start':[0.,1.],'t-end':[1.,2.],'label':[0,0]})
    f=pd.DataFrame({'video-id':['a']*4,'t-start':[0.,0.,1.,5.],'t-end':[1.,1.,2.,6.],
                    'label':[0]*4,'score':[.9,.8,.7,.95],'pbg':[.2,.5,.1,.99]})
    e=ClassAP(gt,f)
    for lam in [0.,.25,1.,4.]:
        scores=f.score.to_numpy()*np.clip(1-f.pbg.to_numpy(),1e-8,1)**lam
        actual=summarize_ap(e,np.ones(len(f),dtype=bool),scores)
        other=f.copy();other['score']=scores
        official=compute_average_precision_detection(gt,other)
        np.testing.assert_allclose(actual['ap'],official[0],atol=1e-14,rtol=0)
        np.testing.assert_array_equal(actual['tp'],official[1])


def test_auc_ties_and_degenerate_roles():
    assert auc([0,1],[.2,.8])==1.
    assert auc([0,1],[.2,.2])==.5
    assert auc([1,1],[.2,.8]) is None

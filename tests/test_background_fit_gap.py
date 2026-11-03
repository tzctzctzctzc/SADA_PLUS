import numpy as np
from tools.background_fit_gap import auc, summarize
from tools.audit_background_fit_gap import independent_auc


def test_auc_ties_and_missing_class():
    assert auc([1,0],[.9,.1])==1
    assert auc([1,0],[.1,.9])==0
    assert auc([1,0],[.5,.5])==.5
    assert auc([1,1],[.9,.1]) is None
    labels=np.array([1,0,1,0,1]);scores=np.array([.2,.2,.1,.5,.9])
    expected=np.mean([float(a>b)+.5*float(a==b) for a in scores[labels==1] for b in scores[labels==0]])
    assert auc(labels,scores)==expected
    assert independent_auc(labels,scores)==expected


def test_macro_denominator_and_fixed_budget():
    rows=[dict(video='a',bg=[True,False]*100,scores=[.8]*200,labels=[0]*200,delta=[.1,-.2]*100,pbg={'control':[.9,.1]*100}),dict(video='b',bg=[False],scores=[.9],labels=[1],delta=[-.1],pbg={'control':[.5]})]
    out=summarize(rows,'control')
    assert out['AUC_valid_videos']==1 and out['total_videos']==2 and out['video_macro_AUC']==1
    assert [r['deleted'] for r in out['all_fixed_budgets']]==[1,2,4,10]
    assert all(r['nonBG_removed']==0 and r['deleted_harm_cost_share_proxy_not_AP']==0 for r in out['all_fixed_budgets'])

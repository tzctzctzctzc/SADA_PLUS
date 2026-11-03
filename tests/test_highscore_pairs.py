import numpy as np
from tools.highscore_confusion_diagnosis import matched_pairs


def test_pairs_never_cross_class_bins_or_video_and_require_AP_sign():
    n=40;labels=np.r_[np.zeros(20,int),np.ones(20,int)];scores=np.full(n,.8);length=np.full(n,2.);videos=np.array(['a','b']*20)
    bg=np.arange(n)%4==0;delta=np.where(bg,.01,-.02);delta[4]=0
    features=np.eye(n)
    out=matched_pairs(labels,scores,length,videos,bg,delta,features)
    assert out['pairs']
    for p in out['pairs']:
        b,f=p['BG'],p['foreground'];assert b!=4 and bg[b] and not bg[f]
        assert labels[b]==labels[f] and videos[b]!=videos[f] and delta[b]>0 and delta[f]<0
    assert out==matched_pairs(labels,scores,length,videos,bg,delta,features)


def test_no_foreground_pair_does_not_invent_one():
    out=matched_pairs(np.zeros(3,int),np.array([.5,.6,.7]),np.ones(3),np.array(['a','b','c']),np.ones(3,bool),np.ones(3),np.eye(3))
    assert not out['pairs'] and len(out['unpaired_BG'])==3

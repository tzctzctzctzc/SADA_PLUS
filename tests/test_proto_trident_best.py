import json
from pathlib import Path
import numpy as np
from tools.proto_trident_best import rescore,MANIFEST


def record():
    return dict(uncertainty_scores=np.array([.2,.8]),labels=np.array([0,1]),
        prototype_cosines=np.array([[.6,.2],[.1,.7]],np.float32),branch_classifier_logits=np.array([[1.,0.],[0.,1.]],np.float32),
        source_prototype_ready=np.array([True,True]))


def test_neutral_readout_and_unready_class_preserve_scores():
    r=record();assert np.array_equal(rescore(r,dict(readout='baseline')),r['uncertainty_scores'])
    r['source_prototype_ready'][:]=False
    assert np.allclose(rescore(r,dict(readout='prototype',temperature=.5,beta=.25)),r['uncertainty_scores'],rtol=0,atol=1e-15)


def test_cosine_and_existing_classifier_use_their_correct_logits():
    r=record()
    for readout,field in [('prototype','prototype_cosines'),('branch_classifier','branch_classifier_logits')]:
        logits=r[field].astype(float)/.5;logits=logits-logits.max(1,keepdims=True)
        q=np.exp(logits);q=q/q.sum(1,keepdims=True);q=q[np.arange(2),r['labels']]
        s=r['uncertainty_scores'];expected=np.exp(-np.logaddexp(0.,-(np.log(s/(1-s))+.25*np.log(q/(1-q)))))
        assert np.array_equal(expected,rescore(r,dict(readout=readout,temperature=.5,beta=.25)))


def test_release_mean_and_all_six_checkpoint_bindings():
    d=json.loads(MANIFEST.read_text(encoding='utf-8'));assert len(d['settings'])==6
    assert abs(np.mean([v['map_percent'] for v in d['settings'].values()])-d['per_shift_mean_percent'])<1e-10
    assert all(len(v['checkpoint_sha256'])==len(v['config_sha256'])==64 for v in d['settings'].values())
    assert d['settings']['S4']['readout']['readout']=='baseline'

import numpy as np
import pandas as pd
import torch
from tools.background_ap_cost import deletion_costs, cost_balanced_bg
from tools.foreground_oracle_ap import ClassAP


def frames(scores, boxes, labels=None):
    n = len(scores)
    return pd.DataFrame({'video-id':['v']*n,'t-start':[b[0] for b in boxes],
        't-end':[b[1] for b in boxes],'label':labels or [0]*n,'score':scores,'pbg':[0.]*n})


def test_critical_and_background_costs():
    gt = frames([1.], [(0, 1)])
    pred = frames([.9, .8], [(2, 3), (0, 1)])
    d = deletion_costs(gt, pred)
    assert np.allclose(d, [.5, -.5])


def test_low_score_replacement_is_kept():
    gt = frames([1.], [(0, 1)])
    pred = frames([.9, .2], [(0, 1), (0, 1)])
    assert np.array_equal(deletion_costs(gt, pred), [0., 0.])


def test_wrong_class_and_empty_class():
    gt = frames([1.], [(0, 1)])
    pred = frames([.9], [(0, 1)], [1])
    assert deletion_costs(gt, pred)[0] == 0


def test_ties_match_independent_official():
    from libs.evaluation.evaluator_anet import compute_average_precision_detection
    gt = frames([1.,1.], [(0,1),(3,4)])
    pred = frames([.8,.8,.8,.8,.2], [(0,1),(2,3),(3,4),(0,1),(3,4)])
    base = compute_average_precision_detection(gt, pred, np.linspace(.1,.5,5))[0].mean()
    delta = deletion_costs(gt,pred)
    for i in range(4):
        ap = compute_average_precision_detection(gt,pred.drop(i).reset_index(drop=True),np.linspace(.1,.5,5))[0].mean()
        assert abs(delta[i] - (ap-base)) < 1e-12


def test_cost_increases_critical_foreground_protection():
    logits = torch.zeros(4, requires_grad=True)
    bg = torch.tensor([0.,0.,1.,1.])
    delta = torch.tensor([-.4,-.1,.3,.1])
    cost_balanced_bg(logits,bg,delta).backward()
    assert logits.grad[0] > logits.grad[1] > 0
    assert logits.grad[2] < logits.grad[3] < 0


def test_zero_cost_fallback_finite_and_uniform():
    logits = torch.zeros(3, requires_grad=True)
    loss = cost_balanced_bg(logits,torch.zeros(3),torch.zeros(3))
    loss.backward()
    assert torch.isfinite(loss) and torch.allclose(logits.grad,torch.full((3,),1/6))

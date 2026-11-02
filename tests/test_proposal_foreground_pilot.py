import torch
from tools.proposal_foreground_pilot import full_candidates,pick


def test_full_video_negative_start_and_subgrid_gt_not_dropped():
    gt=torch.tensor([[-1.,-.94],[1.,1.07]])
    b=full_candidates(torch.empty(0,2),gt,-1.,10.)
    assert torch.equal(b[:2],gt) and len(b)==6
    assert (b[:,1]>b[:,0]).all()


def test_selection_independent_of_input_order():
    ids=['v%d'%i for i in range(30)]
    assert pick(ids,12,'fit')==pick(ids[::-1],12,'fit')

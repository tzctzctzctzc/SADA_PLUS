import numpy as np
import torch
from tools.audit_uda_background import semantic_numpy
from tools.uda_background_math import semantic_targets


def test_independent_semantic_assignment_and_foreground_priority():
    boxes=np.array([[0.,10.],[20.,30.],[50.,60.],[70.,80.]],np.float32);gt=np.array([[0.,1.],[0.,10.],[20.,30.],[70.,70.5]],np.float32);labels=np.array([2,34,11,5])
    expected=semantic_numpy(boxes,gt,labels);actual=semantic_targets(torch.from_numpy(boxes),torch.from_numpy(gt),torch.from_numpy(labels)).numpy()
    assert np.array_equal(expected,actual) and expected.tolist()==[2,11,-1,-1]

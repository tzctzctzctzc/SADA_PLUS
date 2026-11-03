import numpy as np
import torch
from tools.audit_detector_iacs import quality_numpy
from tools.detector_iacs_math import iou_targets


def test_independent_quality_thresholds():
    c=np.eye(10,dtype=np.float32)[:4];g=np.array([[1,1],[1,1],[1,1],[0,0]],np.float32);p=np.array([[1,1],[3,3],[10,10],[1,1]],np.float32)
    assert np.allclose(quality_numpy(c,g,p),iou_targets(torch.from_numpy(c),torch.from_numpy(g),torch.from_numpy(p),True).numpy(),atol=3e-8,rtol=0)

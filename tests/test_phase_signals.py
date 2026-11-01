import numpy as np
from tools.phase_signals import pairwise_iou,seconds
import torch

def test_pairwise_instance_iou():
 a=np.array([[0.,2.],[4.,5.]])
 b=np.array([[1.,3.],[4.,5.]])
 assert np.allclose(pairwise_iou(a,b),[[1/3,0],[0,1]])

def test_physical_center_conversion_and_clipping():
 m=dict(use_temporal_grid_feats=True,feat_stride=16,feat_num_frames=32,fps=32,duration=10)
 assert torch.allclose(seconds(torch.tensor([[-1.,0.],[18.,21.]]),m),torch.tensor([[0.,.5],[9.5,10.]]))

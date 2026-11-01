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


def test_atomic_replacement_does_not_change_saved_snapshot(tmp_path):
 from tools.phase_signals import snapshot_file
 from tools.baseline_protocol import save,read
 entry=tmp_path/'entry';entry.mkdir();save(entry/'source_val_unlabeled.json',{'database':{}})
 source=entry/'latest.pt';torch.save(dict(epoch=5,config={},tensor=torch.tensor([1.])),source)
 job=snapshot_file(source,tmp_path/'out',entry)
 temp=entry/'latest.tmp';torch.save(dict(epoch=10,config={},tensor=torch.tensor([2.])),temp);temp.replace(source)
 assert torch.load(job/'checkpoint.pt')['epoch']==5
 assert torch.load(job/'checkpoint.pt')['tensor'].item()==1
 assert torch.load(source)['epoch']==10
 assert read(job/'snapshot.json')['immutable_atomic_inode']

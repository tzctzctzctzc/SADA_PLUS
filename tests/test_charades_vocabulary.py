"""新数据集词表必须正确，旧厨房默认行为必须保留。"""
from pathlib import Path
from runpy import run_path
import hashlib
import json
import pytest
import torch

MODULE=run_path(str(Path(__file__).resolve().parents[1]/'libs/models/SADA/prototype_capacity.py'))
LABELS=('TAKE','PUT','SIT','WALK','HOLD','STAND','OPEN','DRINK','SMILE','LAUGH')


@pytest.mark.parametrize('index,label',list(enumerate(LABELS)))
def test_charades_index(index,label):
    x=MODULE['evidence_vector'](label,10,'cpu',torch.float32,label_names=LABELS)
    assert int(x.argmax())==index and float(x.sum())==1


def test_old_default_is_unchanged():
    for i,label in enumerate(MODULE['KNOWN_LABELS']):
        x=MODULE['evidence_vector'](label,10,'cpu',torch.float32)
        assert int(x.argmax())==i and float(x.sum())==1
    with pytest.raises(ValueError):MODULE['evidence_vector']('SIT',10,'cpu',torch.float32)


def test_custom_vocabulary_reaches_real_prototype_and_gradient(tmp_path):
    row=dict(proposal_start_sec=16.,proposal_end_sec=36.,source_fpn_levels=[0],candidate_id='a',
        semantic_label='SIT',class_probabilities=[0.,0.,1.]+[0.]*7,participation_selected=True)
    path=tmp_path/'cache.json';path.write_text(json.dumps(dict(label_names=LABELS,by_video={'t':[row]})))
    cfg=dict(variant='control',prepared_cache=str(path),prepared_cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    model=MODULE['build_capacity'](cfg,8,10,[1],[[0,10000]],'cpu')
    assert model.semantic_label_names==LABELS
    s=dict(feats=torch.randn(8,64),segments=torch.tensor([[15.,35.]]),labels=torch.tensor([2]),
        metadata=dict(video_id='s',fps=1.,feat_stride=1.,feat_num_frames=2,crop_start=0,crop_feat_len=64))
    t=dict(feats=torch.randn(8,64),segments=torch.empty(0,2),labels=torch.empty(0,dtype=torch.long),
        metadata=dict(video_id='t',fps=1.,feat_stride=1.,feat_num_frames=2,crop_start=0,crop_feat_len=64))
    sf=[s['feats'][None].requires_grad_()];tf=[t['feats'][None].requires_grad_()]
    out=model([s],[t],sf,tf,5);out['loss'].backward()
    assert out['selected']==1 and out['target_gradient_norm']>0 and tf[0].grad.abs().sum()>0

from copy import deepcopy

import pytest
import torch

from tools.rq4_temporal_probe_core import (
    POLICY, ARMS, TemporalProbe, phase_distribution, paired_distribution,
    real_indices, local_patches, alignment_loss, supervised_loss, source_bank,
    video_batches, check_training_payload, score_predictions, score_routes, train_arms)
from tools.run_rq4_temporal_probe import membership


def payload():
    generator=torch.Generator().manual_seed(9)
    source=dict(x=torch.randn(36,8,9,generator=generator),
        roles=torch.eye(3).repeat(12,1),log_offsets=torch.ones(36,2),
        video_ids=['train_a']*18+['train_b']*18)
    rows=dict(x=torch.randn(14,8,9,generator=generator),video_ids=['heldout_a']*7+['heldout_b']*7,
        candidate_ids=['a']*7+['b']*7,times=torch.linspace(0,6,7).repeat(2),
        selected=torch.ones(14,dtype=torch.bool),
        routes={name:phase_distribution(torch.linspace(0,6,7).repeat(2),[0,6]) for name in ARMS[1:]})
    return source,rows


def test_phase_interpolation_and_outside_abstention():
    q=phase_distribution([-1,0,.25,.5,.75,1,2],[0,1])
    assert torch.allclose(q,torch.tensor([[0,0,0],[1,0,0],[.5,.5,0],[0,1,0],
                                       [0,.5,.5],[0,0,1],[0,0,0]]))


def test_missing_context_is_not_upweighted():
    q=paired_distribution([0,.5,1],[[0,1],None])
    assert torch.allclose(q.sum(-1),torch.full((3,),.5))
    assert not paired_distribution([0,.5],[None,None]).any()


def test_real_grid_does_not_create_short_interval_points():
    assert real_indices([.60,.65],30,16,32,100,7).numel()==0
    assert real_indices([0,1],30,16,32,100,7).tolist()==[0]
    assert len(real_indices([0,10],30,16,32,100,7))==7


def test_visual_patch_has_no_proposal_or_gt_coordinate():
    feats=torch.arange(20.).reshape(2,10)
    x=local_patches(feats,torch.tensor([0,5]))
    assert x.shape==(2,2,9)
    assert x[0,0].tolist()==[0,0,0,0,0,1,2,3,4]
    assert x[1,0].tolist()==list(range(1,10))


def test_membership_is_video_disjoint():
    cfg=dict(src_train_split=['train'],src_val_split=['val'],tgt_train_split=['target'])
    db={k:dict(subset=k) for k in ('train','val','target')}
    assert membership(cfg,db)['heldout']==['val']
    cfg['tgt_train_split']=['val']
    with pytest.raises(ValueError,match='overlap'):membership(cfg,db)


@pytest.mark.parametrize('field', ['gt','segments','labels','quality_truth'])
def test_training_contract_rejects_heldout_annotation_fields(field):
    source,rows=payload();rows[field]=[]
    with pytest.raises(ValueError,match='labels'):check_training_payload(source,rows)


def test_training_contract_rejects_overlap_and_invalid_routes():
    source,rows=payload();check_training_payload(source,rows)
    rows['video_ids'][0]='train_a'
    with pytest.raises(ValueError,match='overlap'):check_training_payload(source,rows)
    rows['video_ids'][0]='heldout_a';rows['routes']['proposal'][0,0]=float('nan')
    with pytest.raises(ValueError,match='routing'):check_training_payload(source,rows)


def test_batches_are_arm_independent_and_fixed():
    ids=['a']*8+['b']*5+['c']*11
    first=video_batches(ids,5)
    assert all(torch.equal(a,b) for a,b in zip(first,video_batches(ids,5)))
    assert all(len(set(ids[i] for i in batch))==2 for batch in first)


def test_zero_auxiliary_is_exactly_zero_gradient():
    model=TemporalProbe(8);x=torch.randn(6,8,9)
    bank=torch.nn.functional.normalize(torch.randn(3,4,128),dim=-1)
    loss=alignment_loss(model(x)[0],bank,torch.zeros(6,3))
    assert loss.item()==0
    loss.backward();assert model.project.weight.grad.norm()==0


def test_auxiliary_reaches_shared_visual_encoder_not_bank():
    torch.manual_seed(6);model=TemporalProbe(8)
    bank=torch.randn(3,4,128,requires_grad=True)
    loss=alignment_loss(model(torch.randn(6,8,9))[0],bank,torch.eye(3).repeat(2,1))
    loss.backward()
    assert torch.isfinite(model.project.weight.grad).all()
    assert model.project.weight.grad.norm()>0
    assert bank.grad is None


def test_bank_uses_only_source_and_is_deterministic():
    source,_=payload();model=TemporalProbe(8)
    bank=source_bank(model,source['x'],source['roles'],'cpu')
    assert bank.shape==(3,4,128) and not bank.requires_grad
    assert torch.equal(bank,source_bank(model,source['x'],source['roles'],'cpu'))
    assert torch.allclose(bank.norm(dim=-1),torch.ones(3,4),atol=1e-5)


def test_supervised_probe_has_offset_and_role_gradients():
    source,_=payload();model=TemporalProbe(8)
    supervised_loss(model,source['x'],source['roles'],source['log_offsets']).backward()
    assert model.offset.weight.grad.norm()>0
    assert model.role.weight.grad.norm()>0


def test_perfect_boundary_scoring_and_common_support():
    _,rows=payload()
    truth=dict(roles=rows['routes']['proposal'].tolist(),candidates=[dict(candidate_id='a',
        video_id='heldout_a',row_indices=list(range(7)),gt=[0,6],duration=10)])
    distances=torch.stack((rows['times'],6-rows['times']),-1)
    pred=dict(roles=rows['routes']['proposal'],log_offsets=distances.log1p())
    result=score_predictions(pred,rows,truth)
    assert result['mean_iou']==pytest.approx(1)
    assert result['normalized_endpoint_error']==pytest.approx(0,abs=1e-6)
    assert result['role_accuracy']==1
    scores=score_routes(rows,truth)
    assert scores['paired_common_support']['timelens2_8b']['proposal']==1
    assert scores['paired_common_support']['timelens2_8b']['teacher']==1


def test_evaluation_labels_cannot_change_training(monkeypatch):
    # 缩短仅用于测试，正式入口要求完整POLICY；不读取任何评分GT。
    monkeypatch.setitem(POLICY,'warmup_steps',2)
    monkeypatch.setitem(POLICY,'adaptation_steps',2)
    source,rows=payload();saved={}
    result=train_arms(source,rows,'cpu',lambda *a:None,
        lambda arm,model,bank,history,pred:saved.update({arm:deepcopy(history)}))
    assert set(result)==set(ARMS)
    assert all(h['heldout_gt_loss']==0 for values in saved.values() for h in values)
    assert all(h['shared_visual_gradient']==0 for h in saved['none'])
    assert any(h['shared_visual_gradient']>0 for h in saved['proposal'])
    # 同路由的两教师与proposal完全同轨迹，证明无隐藏按arm改变RNG/预算。
    assert torch.equal(result['proposal']['log_offsets'],result['timelens2_8b']['log_offsets'])
    assert torch.equal(result['proposal']['log_offsets'],result['timeple_8b']['log_offsets'])
    before=result['none']['log_offsets'].clone()
    truth=dict(roles=torch.eye(3)[torch.arange(14)%3].tolist(),candidates=[])
    score_predictions(result['none'],rows,truth)
    assert torch.equal(before,result['none']['log_offsets'])

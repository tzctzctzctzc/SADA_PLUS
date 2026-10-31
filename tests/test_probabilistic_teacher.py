from copy import deepcopy
import json
from pathlib import Path
from runpy import run_path

import pytest
import torch

from libs.models.SADA.probabilistic_temporal import (TemporalVarianceHead,specification,
    shifted_gaussian_nll,gaussian_kl,target_losses,masked_videos,source_likelihood)
from tools.baseline_protocol import CODE


def test_probability_losses_match_torch_distributions_and_stop_teacher_gradients():
    m=torch.tensor([.2,.8],requires_grad=True);v=torch.tensor([.3,.7],requires_grad=True)
    teacher_m=torch.tensor([.1,.9],requires_grad=True);teacher_v=torch.tensor([.4,.5],requires_grad=True)
    actual=gaussian_kl(m,v,teacher_m,teacher_v)
    expected=torch.distributions.kl_divergence(torch.distributions.Normal(teacher_m.detach(),teacher_v.detach().sqrt()),
                                             torch.distributions.Normal(m,v.sqrt()))
    torch.testing.assert_close(actual,expected)
    actual.sum().backward();assert m.grad.abs().sum()>0 and v.grad.abs().sum()>0
    assert teacher_m.grad is None and teacher_v.grad is None
    assert torch.allclose(gaussian_kl(m.detach(),v.detach(),m.detach(),v.detach()),torch.zeros_like(m))
    nll=shifted_gaussian_nll(m.detach(),v.detach(),teacher_m.detach())
    reference=-torch.distributions.Normal(m.detach(),v.detach().sqrt()).log_prob(teacher_m.detach())
    assert torch.allclose(nll-reference,(nll-reference)[0].expand_as(nll))


def packet(logits,offsets,variance):
    points=torch.tensor([[1.,0.,4.,1.],[2.,0.,4.,1.],[3.,0.,4.,1.],[4.,0.,4.,1.]])
    return dict(logits=[logits],offsets=[offsets],variance=[variance],
                masks=[torch.ones(1,4,dtype=torch.bool)],points=[points])


def target_example(label='TAKE',scores=None):
    video=dict(feats=torch.ones(8,10),labels=torch.empty(0,dtype=torch.long),segments=torch.empty(0,2),
        metadata=dict(video_id='t',fps=1.,feat_num_frames=0,feat_stride=1,crop_start=0,crop_feat_len=10))
    row=dict(proposal_start_sec=0.,proposal_end_sec=8.,source_fpn_levels=[0],candidate_id='fixed',
        semantic_label=label,class_probabilities=scores or [.9]+[.1/9]*9,participation_selected=True)
    student=packet(torch.zeros(1,4,10,requires_grad=True),torch.full((1,4,2),2.5,requires_grad=True),
                   torch.full((1,4,2),.2,requires_grad=True))
    logits=torch.full((1,4,10),-3.);logits[:,:,0]=3.;logits.requires_grad_()
    teacher=packet(logits,torch.full((1,4,2),2.,requires_grad=True),torch.full((1,4,2),.1,requires_grad=True))
    return video,row,student,teacher


def test_target_losses_update_actual_classification_and_localization_without_teacher_grad():
    video,row,student,teacher=target_example()
    sem,loc,stats=target_losses(student,teacher,[video],{'t':[row]},True)
    assert stats['accepted']==1 and stats['native_points']==4 and sem>0 and loc>0
    (sem+loc).backward()
    assert student['logits'][0].grad.abs().sum()>0 and student['offsets'][0].grad.abs().sum()>0
    assert student['variance'][0].grad.abs().sum()>0
    assert teacher['logits'][0].grad is None and teacher['offsets'][0].grad is None
    assert teacher['variance'][0].grad is None
    # 未被VLM证实的其他类别没有负类梯度。
    assert student['logits'][0].grad[:,:,1:].abs().sum()==0


@pytest.mark.parametrize('label,scores',[
    ('UNKNOWN_ACTION',None),('BACKGROUND',None),('ABSTAIN',None),
    ('PUT',[.1,.9]+[0.]*8),('TAKE',[.5,.5]+[0.]*8)])
def test_unknown_conflicts_and_ties_are_not_forced_to_foreground(label,scores):
    video,row,student,teacher=target_example(label,scores)
    sem,loc,stats=target_losses(student,teacher,[video],{'t':[row]},True)
    assert stats['routable']==1 and stats['accepted']==0 and sem==loc==0
    _,_,plain=target_losses(student,teacher,[video],{'t':[row]},False)
    assert plain['accepted']==1


def test_teacher_only_does_not_read_vlm_labels_or_target_gt():
    video,row,student,teacher=target_example()
    class GeometryOnly(dict):
        def get(self,key,*args):
            if key in ('semantic_label','class_probabilities'):raise AssertionError('EMA组读取VLM语义')
            return super().get(key,*args)
    video['labels']=video['segments']='禁止读取目标GT'
    sem,loc,stats=target_losses(student,teacher,[video],{'t':[GeometryOnly(row)]},False)
    assert sem>0 and loc>0 and stats['accepted']==1


def test_cache_visibility_and_native_point_masks_are_respected():
    video,row,student,teacher=target_example();row['proposal_end_sec']=20.
    assert target_losses(student,teacher,[video],{'t':[row]},True)[2]['routable']==0
    row['proposal_end_sec']=8.;student['masks'][0][:]=False
    assert target_losses(student,teacher,[video],{'t':[row]},True)[2]['routable']==0


def test_mask_preserves_time_grid_metadata_and_uses_its_own_random_stream():
    video,*_=target_example();before=video['feats'].clone();state=torch.get_rng_state().clone()
    generator=torch.Generator().manual_seed(123)
    output=masked_videos([video],generator)[0]
    assert output['metadata']==video['metadata'] and output['feats'].shape==before.shape
    assert torch.equal(video['feats'],before) and torch.equal(torch.get_rng_state(),state)
    assert output['labels'].numel()==0 and output['segments'].numel()==0


def test_true_sada_optional_head_keeps_old_detector_initialization_and_source_losses(monkeypatch):
    from libs.models.SADA.meta_archs import SADA
    small=run_path(str(CODE/'tests/test_corrected_baseline.py'))['small_config']
    native=torch.tensor
    def cpu_copy(*args,**kwargs):
        value=native(*args,**kwargs)
        return value.clone() if value.requires_grad and value.device.type=='cpu' else value
    monkeypatch.setattr(torch,'tensor',cpu_copy)
    cfg=small();cfg['model']['adv_method']=['None'];new=deepcopy(cfg)
    new['probabilistic_teacher']=dict(variant='source',specification=specification())
    torch.manual_seed(123);old=SADA(cfg,'cpu')
    torch.manual_seed(123);model=SADA(new,'cpu')
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in old.state_dict().items())
    source=dict(feats=torch.randn(8,64),segments=torch.tensor([[10.,35.]]),labels=torch.tensor([0]),
        metadata=dict(video_id='s',fps=1.,feat_stride=1.,feat_num_frames=0,crop_start=0,crop_feat_len=64))
    target=deepcopy(source);target['segments']=torch.empty(0,2);target['labels']=torch.empty(0,dtype=torch.long)
    torch.manual_seed(777);a=old(video_list_src=[source],video_list_tgt=[target],mode='train',curr_epoch=5)[0]
    torch.manual_seed(777);b=model(video_list_src=[source],video_list_tgt=[target],mode='train',curr_epoch=5)[0]
    for key in ('cls_loss','reg_loss','downstream_loss'):assert torch.equal(a[key],b[key])
    nll,count=source_likelihood(model,model.pt_packet['source'],[source]);assert count>0 and nll>0
    (b['downstream_loss']+.1*nll).backward()
    assert model.pt_uncertainty.layers[-1].weight.grad.abs().sum()>0
    restored=SADA(new,'cpu');restored.load_state_dict(model.state_dict(),strict=True)
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items())
    model.eval()
    with torch.no_grad():raw=model(video_list_src=[target],video_list_tgt=[target],mode='pt_raw')
    assert len(raw['target']['variance'])==6
    assert all(not value.requires_grad for value in raw['target']['variance'])


def test_empty_source_positive_region_is_a_legal_zero():
    _,_,student,_=target_example()
    class Model:
        def label_points(self,*args):return [torch.zeros(4,10)],[torch.zeros(4,2)]
    video=dict(segments=torch.empty(0,2),labels=torch.empty(0,dtype=torch.long))
    loss,count=source_likelihood(Model(),student,[video]);assert count==0 and loss==0


def test_source_configuration_and_real_target_pairing(monkeypatch,tmp_path):
    from tools import run_probabilistic_teacher as driver
    old=dict(seed=123,init_rand_seed=123,loader={'batch_size':2},
        model={'adv_method':['conditional_rev']},baseline=dict(shift='S1',regime='uda',code={},tmpdir='/old',
        target_manifest='/frozen-target',epoch_budget={'iterations_per_epoch':53}),opt={'epochs':50})
    cache=tmp_path/'cache.json';cache.write_text('{}');control=tmp_path/'control';control.mkdir()
    (control/'config.json').write_text(json.dumps(dict(prototype_capacity=dict(prepared_cache=str(cache),prepared_cache_sha256='frozen'))))
    monkeypatch.setattr(driver,'CONTROL',control)
    configs={v:driver.configuration(old,v,{},'/new') for v in driver.VARIANTS}
    for cfg in configs.values():driver.assert_config(cfg,old)
    assert configs['source']['baseline']['regime']=='so'
    assert configs['ema']['baseline']['regime']==configs['vlm']['baseline']['regime']=='uda'
    assert all(c['baseline']['target_manifest']=='/frozen-target' for c in configs.values())
    bad=deepcopy(configs['vlm']);bad['loader']['batch_size']=4
    with pytest.raises(ValueError):driver.assert_config(bad,old)

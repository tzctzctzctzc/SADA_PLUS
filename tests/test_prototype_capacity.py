import ast
from copy import deepcopy
import hashlib
import json
import subprocess

import pytest
import torch

from libs.models.SADA.prototype_capacity import build_capacity,specification,RQ3ProposalPrototype
from tools.baseline_protocol import CODE
from tools.run_prototype_capacity import assert_pair


def configuration(tmp_path,variant):
    row=dict(proposal_start_sec=16.,proposal_end_sec=36.,source_fpn_levels=[0],candidate_id='a',
        semantic_label='TAKE',class_probabilities=[.9]+[.1/9]*9,participation_selected=True)
    path=tmp_path/'prepared.json';path.write_text(json.dumps(dict(by_video={'t':[row]})))
    return dict(variant=variant,prepared_cache=str(path),prepared_cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def videos():
    source=dict(feats=torch.randn(8,64),segments=torch.tensor([[15.,35.]]),labels=torch.tensor([0]),
        metadata=dict(video_id='s',fps=1.,feat_stride=1.,feat_num_frames=2,crop_start=0,crop_feat_len=64))
    target=dict(feats=torch.randn(8,64),segments=torch.empty(0,2),labels=torch.empty(0,dtype=torch.long),
        metadata=dict(video_id='t',fps=1.,feat_stride=1.,feat_num_frames=2,crop_start=0,crop_feat_len=64))
    return source,target


@pytest.mark.parametrize('variant',['control','wide_deep','wide_only'])
def test_branch_shapes_gradient_optimizer_and_strict_restore(tmp_path,variant):
    cfg=configuration(tmp_path,variant);spec=specification(variant)
    torch.manual_seed(123);before=torch.get_rng_state().clone()
    model=build_capacity(cfg,8,97,[1],[[0,10000]],'cpu')
    assert torch.equal(before,torch.get_rng_state())
    assert model.branch.classifier.in_features==spec['embedding_dim'] and len(model.branch.encoder.layers)==spec['layers']
    s,t=videos();sf=[s['feats'][None].requires_grad_()];tf=[t['feats'][None].requires_grad_()]
    output=model([s],[t],sf,tf,5)
    assert output['selected']>0 and output['source_loss']>0 and output['alignment_loss']>0 and output['target_gradient_norm']>0
    opt=torch.optim.AdamW(model.parameters(),lr=.0001)
    initial=model.branch.input_projection.weight.detach().clone()
    output['loss'].backward();assert sf[0].grad.abs().sum()>0 and tf[0].grad.abs().sum()>0
    opt.step();assert not torch.equal(initial,model.branch.input_projection.weight)
    restored=build_capacity(cfg,8,97,[1],[[0,10000]],'cpu');restored.load_state_dict(model.state_dict(),strict=True)
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items())


def test_control_is_old_prototype_numerically_identical(tmp_path):
    cfg=configuration(tmp_path,'control')
    model=build_capacity(cfg,8,97,[1],[[0,10000]],'cpu')
    legacy=RQ3ProposalPrototype(8,97,json.loads((tmp_path/'prepared.json').read_text())['by_video'],[1],[[0,10000]],0.)
    assert all(torch.equal(v,legacy.state_dict()[k]) for k,v in model.state_dict().items())
    s,t=videos();sf=[s['feats'][None].requires_grad_()];tf=[t['feats'][None].requires_grad_()]
    a=model([s],[t],sf,tf,5);b=legacy([s],[t],sf,tf,5)
    for key in a:
        assert torch.equal(torch.as_tensor(a[key]),torch.as_tensor(b[key]))


def test_copied_encoder_and_targets_match_frozen_legacy_source():
    original=subprocess.check_output(['git','show','6da910b78c17953924821ed49606dad7dce30524:libs/models/SADA/proposal_semantic.py'],cwd=CODE,text=True)
    current=(CODE/'libs/models/SADA/prototype_capacity.py').read_text(encoding='utf-8')
    def selected(text):
        tree=ast.parse(text)
        return {n.name:ast.dump(n) for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
    a,b=selected(original),selected(current)
    for name in ('VLMProposalCoAdaptationBranch','source_gt_proposals','cached_proposals_for_videos','_sample_sequence','evidence_vector'):
        assert a[name]==b[name]


def test_pairing_rejects_learning_rate_or_seed_changes(tmp_path):
    from tools.run_prototype_capacity import paired_config
    base=dict(seed=123,loader={'batch_size':2},opt={'learning_rate':.0001},baseline=dict(regime='uda',code={'commit':'base'},tmpdir='/old'))
    prepared=tmp_path/'prepared.json';prepared.write_text('{}')
    cfg=paired_config(base,'wide_deep',prepared,{'commit':'new'},tmp_path)
    assert_pair(cfg,base)
    for path,value in [('seed',6),('opt',{'learning_rate':.0002})]:
        bad=deepcopy(cfg);bad[path]=value
        with pytest.raises(ValueError):assert_pair(bad,base)


@pytest.mark.parametrize('variant',['control','wide_deep','wide_only'])
def test_true_sada_integration_keeps_detector_initialization_and_source_loss(tmp_path,monkeypatch,variant):
    from runpy import run_path
    small_config=run_path(str(CODE/'tests/test_corrected_baseline.py'))['small_config']
    from libs.models.SADA.meta_archs import SADA
    from libs.models.SADA.optimizer import make_optimizer
    native=torch.tensor
    def cpu_copy(*args,**kwargs):
        value=native(*args,**kwargs)
        return value.clone() if value.requires_grad and value.device.type=='cpu' else value
    monkeypatch.setattr(torch,'tensor',cpu_copy)
    cfg=small_config();cfg['prototype_capacity']=configuration(tmp_path,variant)
    plain=deepcopy(cfg);plain.pop('prototype_capacity')
    torch.manual_seed(123);base=SADA(plain,'cpu')
    torch.manual_seed(123);aux=SADA(cfg,'cpu')
    assert all(torch.equal(v,aux.state_dict()[k]) for k,v in base.state_dict().items())
    opt=make_optimizer(aux,cfg['opt'],.0001,exclude_params=['disc_dom_adv','disc_dom_conditional_adv','learnable_embeddings'])
    ids={id(p) for g in opt.param_groups for p in g['params']}
    assert all(id(p) in ids for p in aux.rq3_capacity.parameters())
    s,t=videos()
    torch.manual_seed(777);old=base(video_list_src=[s],video_list_tgt=[t],mode='train',curr_epoch=5)[0]
    torch.manual_seed(777);new=aux(video_list_src=[s],video_list_tgt=[t],mode='train',curr_epoch=5)[0]
    assert torch.equal(old['cls_loss'],new['cls_loss']) and torch.equal(old['reg_loss'],new['reg_loss'])
    assert new['rq3_target_gradient_norm']>0
    torch.testing.assert_close(new['downstream_loss'],old['downstream_loss']+new['rq3_loss'])
    new['downstream_loss'].backward()
    assert aux.rq3_capacity.branch.input_projection.weight.grad.abs().sum()>0

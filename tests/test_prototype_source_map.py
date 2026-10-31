from copy import deepcopy
import hashlib,json,ast
import pytest,torch
from libs.models.SADA.prototype_capacity import build_capacity,specification,RQ3ProposalPrototype
from tools.baseline_protocol import CODE,POLICY
from tools.run_prototype_source_map import paired_config,assert_pair,validate_probe,validate_aux


@pytest.mark.parametrize('variant',['control'])

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

@pytest.mark.parametrize('variant',['control'])

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

@pytest.mark.parametrize('variant',['control'])

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


def test_pairing_preserves_map_policy_and_real_target_stream(tmp_path):
    p=tmp_path/'prepared.json';p.write_text('{}')
    base=dict(seed=123,loader={'batch_size':2},model={'adv_method':['conditional_rev']},opt={'learning_rate':.0001},baseline=dict(regime='uda',policy=POLICY,code={},tmpdir='/old'))
    cfg=paired_config(base,p,{'commit':'new'},tmp_path)
    assert_pair(cfg,base)
    assert cfg['baseline']['regime']=='uda' and cfg['model']['adv_method']==['None']
    assert cfg['baseline']['policy']['selection']=='source_validation_map_0.1_to_0.5'
    for field,value in [('seed',6),('opt',{'learning_rate':.0002})]:
        bad=deepcopy(cfg);bad[field]=value
        with pytest.raises(ValueError):assert_pair(bad,base)
    bad=deepcopy(cfg);bad['prototype_capacity']['specification']['embedding_dim']=256
    with pytest.raises(ValueError):assert_pair(bad,base)


def test_acceptance_rejects_zero_target_gradient_and_wrong_initialization():
    good=dict(rq3_target_gradient_norm=.01,rq3_alignment_loss=.2,rq3_source_loss=.3,rq3_selected=2,rq3_loss=.32)
    validate_aux(good)
    for field in ('rq3_target_gradient_norm','rq3_alignment_loss','rq3_selected'):
        bad=dict(good);bad[field]=0
        with pytest.raises(ValueError):validate_aux(bad)
    with pytest.raises(ValueError):validate_probe(dict(status='passed',initial_detector_sha='wrong'),'expected')


def test_source_map_selection_unchanged_by_branch():
    text=(CODE/'tools/baseline_engine.py').read_text(encoding='utf-8')
    train=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='train')
    value=ast.unparse(train)
    assert 'source_validation_map(ctx, epoch' in value and 'score > best' in value
    assert 'source_validation_loss' in value and 'row['auxiliary']' in value

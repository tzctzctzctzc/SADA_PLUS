from copy import deepcopy
from runpy import run_path
import pytest
import torch
from torch.nn import functional as F

from libs.models.SADA.prototype_capacity import build_capacity
from libs.models.SADA.prototype_methods import StableMultiPrototypeBranch,revision_specification
from tools.baseline_protocol import CODE
from tools.prototype_stable_bank import spherical_clusters,balanced_cap,select_source_segments

h=run_path(str(CODE/'tests/test_prototype_capacity.py'))


def test_stable_warmup_is_exact_original_single_center(tmp_path):
    torch.manual_seed(123);s,t=h['videos']();sf=[s['feats'][None].requires_grad_()];tf=[t['feats'][None].requires_grad_()]
    a=build_capacity(h['configuration'](tmp_path,'control'),8,10,[1],[[0,10000]],'cpu')
    b=build_capacity(h['configuration'](tmp_path,'multi_stable'),8,10,[1],[[0,10000]],'cpu')
    assert all(torch.equal(v,dict(b.branch.named_parameters())[k]) for k,v in a.branch.named_parameters())
    for epoch in range(1,5):
        x=a([s],[t],sf,tf,epoch);y=b([s],[t],sf,tf,epoch)
        for key in ['source_loss','alignment_loss','loss','target_gradient_norm']:assert torch.equal(x[key],y[key])
        assert torch.equal(a.branch.source_prototypes,b.branch.source_prototypes[:,0])
        assert not b.branch.source_prototype_ready[:,1:].any()


def test_weights_change_only_target_term_and_target_gradient(tmp_path):
    s,t=h['videos']();sf=[s['feats'][None].requires_grad_()];tf=[t['feats'][None].requires_grad_()];outs=[]
    for variant in ['control','weight_020','weight_040']:
        model=build_capacity(h['configuration'](tmp_path,variant),8,10,[1],[[0,10000]],'cpu')
        outs.append(model([s],[t],sf,tf,5))
    for out,weight in zip(outs,[.1,.2,.4]):
        assert torch.equal(out['source_loss'],outs[0]['source_loss'])
        assert torch.equal(out['alignment_loss'],outs[0]['alignment_loss'])
        torch.testing.assert_close(out['loss'],out['source_loss']+weight*out['alignment_loss'],rtol=0,atol=0)
        torch.testing.assert_close(out['target_gradient_norm'],outs[0]['target_gradient_norm']*(weight/.1))


def test_source_clusters_represent_real_supported_modes_without_rng_changes():
    torch.manual_seed(99);base=torch.eye(128)[:3];vectors=[];videos=[]
    for k in range(3):
        for i in range(12):vectors.append(base[k]+torch.randn(128)*.002);videos.append('v'+str(i%3))
    x=torch.stack(vectors);before=torch.get_rng_state().clone();bank=spherical_clusters(x,videos)
    assert torch.equal(before,torch.get_rng_state());assert len(bank['centers'])==3
    assert sorted(bank['counts'].tolist())==[12,12,12] and bank['videos'].min()==3
    again=spherical_clusters(x,videos)
    assert torch.equal(bank['centers'],again['centers'])


def test_single_video_cannot_fake_supported_multiple_modes():
    x=torch.eye(128)[:3].repeat_interleave(12,0)
    result=spherical_clusters(x,['same-video']*len(x))
    assert len(result['centers'])==1 and result['fallback']


def test_duplicate_vectors_do_not_create_three_centers():
    x=torch.ones(36,128);result=spherical_clusters(x,['a','b']*18)
    assert len(result['centers'])==1


def test_two_supported_modes_backoff_from_three():
    x=torch.eye(128)[:2].repeat_interleave(12,0)
    result=spherical_clusters(x,['a','b']*12)
    assert len(result['centers'])==2


def test_video_balancing_prevents_single_long_video_dominating():
    rows=[dict(video_id='a',i=i) for i in range(100)]+[dict(video_id='b',i=i) for i in range(100)]
    out=balanced_cap(rows,20);assert sum(x['video_id']=='a' for x in out)==10


def test_source_selection_does_not_mutate_original_video():
    video=dict(segments=torch.tensor([[float(i),float(i+1)] for i in range(30)]),labels=torch.zeros(30,dtype=torch.long),metadata=dict(video_id='v'))
    first=select_source_segments(video,4);second=select_source_segments(video,4)
    assert len(first['labels'])==4 and len(video['labels'])==30
    assert torch.equal(first['segments'],second['segments'])


def test_duplicate_annotations_do_not_fake_cluster_support():
    video=dict(segments=torch.tensor([[1.,2.]]*30),labels=torch.zeros(30,dtype=torch.long),metadata=dict(video_id='v'))
    assert len(select_source_segments(video)['labels'])==1


def test_no_random_filling_after_install_and_strict_restore():
    b=StableMultiPrototypeBranch(8,10)
    bank=spherical_clusters(torch.eye(128)[:1].repeat(20,1),['a','b']*10)
    b.install_source_bank({0:bank});assert b.source_prototype_ready[0].sum()==1
    src=F.normalize(torch.randn(10,128),dim=-1);gt=F.one_hot(torch.zeros(10,dtype=torch.long),10).float()
    b._update_source_prototypes(src,gt);assert b.source_prototype_ready[0].sum()==1
    restored=StableMultiPrototypeBranch(8,10);restored.load_state_dict(b.state_dict(),strict=True)
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in b.state_dict().items())
    with pytest.raises(ValueError):b.install_source_bank({0:bank})


def test_target_only_matches_its_class_and_never_updates_centers():
    b=StableMultiPrototypeBranch(8,10);bank=spherical_clusters(torch.eye(128)[:2].repeat_interleave(12,0),['a','b']*12)
    b.install_source_bank({0:bank});before=b.source_prototypes.clone()
    target=dict(embedding=F.normalize(torch.randn(1,128),dim=-1).requires_grad_());empty=dict(embedding=torch.empty(0,128))
    gt=torch.empty(0,10);selection=dict(labels=torch.tensor([0]),selected=torch.tensor([True]))
    loss,_,_=b.prototype_alignment(empty,gt,target,selection);loss.backward()
    assert torch.equal(before,b.source_prototypes) and b.target_prototype_counts[0].sum()==1
    assert target['embedding'].grad.abs().sum()>0
    bad=dict(labels=torch.tensor([1]),selected=torch.tensor([True]));loss,_,_=b.prototype_alignment(empty,gt,target,bad)
    assert loss==0 and not b.source_prototype_ready[1].any()


@pytest.mark.parametrize('variant',['multi_stable','weight_020','weight_040'])
def test_actual_sada_optimizer_and_detection_loss_are_paired(tmp_path,monkeypatch,variant):
    # 复用真实SADA小配置检查共享检测器/优化器；Linux测试中不替换任何CUDA训练代码。
    from libs.models.SADA.meta_archs import SADA
    from libs.models.SADA.optimizer import make_optimizer
    small=run_path(str(CODE/'tests/test_corrected_baseline.py'))['small_config'];native=torch.tensor
    def leaf_copy(*a,**kw):
        value=native(*a,**kw);return value.clone() if value.requires_grad and value.device.type=='cpu' else value
    monkeypatch.setattr(torch,'tensor',leaf_copy)
    cfg=small();cfg['prototype_capacity']=h['configuration'](tmp_path,variant);plain=deepcopy(cfg);plain.pop('prototype_capacity')
    torch.manual_seed(123);base=SADA(plain,'cpu');torch.manual_seed(123);model=SADA(cfg,'cpu')
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in base.state_dict().items())
    opt=make_optimizer(model,cfg['opt'],.0001,exclude_params=['disc_dom_adv','disc_dom_conditional_adv','learnable_embeddings'])
    ids={id(p) for g in opt.param_groups for p in g['params']};assert all(id(p) in ids for p in model.rq3_capacity.parameters())
    if variant=='multi_stable':
        bank=spherical_clusters(torch.eye(128)[:3].repeat_interleave(12,0),['a','b']*18)
        model.rq3_capacity.branch.install_source_bank({0:bank})
    s,t=h['videos']();torch.manual_seed(777);a=base(video_list_src=[s],video_list_tgt=[t],mode='train',curr_epoch=5)[0]
    torch.manual_seed(777);b=model(video_list_src=[s],video_list_tgt=[t],mode='train',curr_epoch=5)[0]
    assert torch.equal(a['cls_loss'],b['cls_loss']) and torch.equal(a['reg_loss'],b['reg_loss'])
    assert b['rq3_target_gradient_norm']>0
    opt.zero_grad();b['downstream_loss'].backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.backbone_source.parameters())
    opt.step()
    restored=SADA(cfg,'cpu');restored.load_state_dict(model.state_dict(),strict=True)
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items())


def test_actual_source_bank_path_preserves_rng_parameters_and_membership(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import json
    from libs.models.SADA.meta_archs import SADA
    from tools.baseline_engine import tensor_digest
    from tools.prototype_stable_bank import initialize_source_bank,rng_state
    small=run_path(str(CODE/'tests/test_corrected_baseline.py'))['small_config'];native=torch.tensor
    def leaf_copy(*a,**kw):
        value=native(*a,**kw);return value.clone() if value.requires_grad and value.device.type=='cpu' else value
    monkeypatch.setattr(torch,'tensor',leaf_copy)
    cfg=small();cfg['prototype_capacity']=h['configuration'](tmp_path,'multi_stable')
    source=tmp_path/'source.json';source.write_text(json.dumps(dict(database={
        'a':dict(subset='train'),'b':dict(subset='train'),'v':dict(subset='val')})))
    cfg.update(src_train_split=['train'],src_val_split=['val'],dataset=dict(json_file=str(source),feat_folder=str(tmp_path),file_ext='.npz'))
    model=SADA(cfg,'cpu');samples=[]
    for vid in ['a','b']:
        s,_=h['videos']();s['metadata']['video_id']=vid
        s['segments']=torch.stack([torch.tensor([float(i+8),float(i+14)]) for i in range(20)])
        s['labels']=torch.zeros(20,dtype=torch.long);samples.append(s)
        (tmp_path/(vid+'.npz')).write_bytes(b'feature identity fixture')
    class SourceDataset:
        data_list=[dict(id='a'),dict(id='b')]
        def __iter__(self):return iter(samples)
    def only_source(config,path,splits,training):
        assert path==str(source) and splits==['train'] and not training
        return SourceDataset()
    monkeypatch.setattr('tools.baseline_engine.dataset',only_source)
    ctx=SimpleNamespace(cfg=cfg,model=model,generators={'source':torch.Generator().manual_seed(123)})
    before=tensor_digest(rng_state(ctx));parameters=tensor_digest(dict(model.named_parameters()))
    initialize_source_bank(ctx,tmp_path)
    assert tensor_digest(rng_state(ctx))==before and tensor_digest(dict(model.named_parameters()))==parameters
    proof=json.loads((tmp_path/'source_bank.json').read_text())
    assert proof['source_videos']==['a','b'] and not proof['target_data_loaded'] and not proof['source_validation_loaded']
    assert bool(model.rq3_capacity.branch.source_bank_initialized)


def test_configuration_rejects_other_changes():
    from tools.run_prototype_stable_weight import make_config,assert_pair
    ref=dict(seed=123,loader=dict(batch_size=2),baseline=dict(code={'commit':'old'},tmpdir='/old'),
        prototype_capacity=dict(variant='control',specification={'variant':'control'}),opt=dict(lr=.001))
    cfg=make_config(ref,'weight_020',{'commit':'new'},'/new');assert_pair(cfg,ref)
    cfg['opt']['lr']=.1
    with pytest.raises(ValueError):assert_pair(cfg,ref)

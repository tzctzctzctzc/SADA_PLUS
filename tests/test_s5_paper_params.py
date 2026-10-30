from copy import deepcopy
import pytest

from libs.models.SADA.prototype_capacity import specification
from tools.run_s5_paper_params import changed_config,assert_declared_changes,validate_probe,LEVELS,THRESHOLD
from tools.run_prototype_capacity import assert_pair


def old_config(method):
    cfg=dict(seed=123,init_rand_seed=123,loader=dict(batch_size=2),
        model=dict(pl_threshold=.3,**{'weight_level_%d'%i:w for i,w in enumerate([.6,.1,1.,.6,.2,.4])}),
        opt=dict(learning_rate=.0001,epochs=50),src_train_split=['ek55_dark_training'],tgt_train_split=['ek100_dark_training'],
        baseline=dict(shift='S5',regime='uda',code={'commit':'old'},tmpdir='/old',policy={'patience_epochs':15}))
    if method=='prototype':cfg['prototype_capacity']=dict(variant='control',specification=specification('control'),
                                                        prepared_cache='/frozen.json',prepared_cache_sha256='same')
    return cfg


def test_only_declared_s5_parameters_change_and_pairing_survives():
    configs={}
    for method in ('uda','prototype'):
        old=old_config(method);before=deepcopy(old);new=changed_config(old,{'commit':'new'},'/new')
        assert old==before;assert_declared_changes(new,old,method);configs[method]=new
        assert [new['model']['weight_level_%d'%i] for i in range(6)]==list(LEVELS)
        assert new['model']['pl_threshold']==THRESHOLD==.5
    assert_pair(configs['prototype'],configs['uda'])


@pytest.mark.parametrize('change',['seed','batch','lr','direction','threshold','level','cache','width','aux_weight','patience','disc_width'])
def test_unapproved_changes_are_rejected(change):
    old=old_config('prototype');new=changed_config(old,{'commit':'new'},'/new')
    if change=='seed':new['seed']=6
    elif change=='batch':new['loader']['batch_size']=4
    elif change=='lr':new['opt']['learning_rate']=.0002
    elif change=='direction':new['src_train_split']=['ek100_dark_training']
    elif change=='threshold':new['model']['pl_threshold']=.3
    elif change=='level':new['model']['weight_level_1']=.3
    elif change=='cache':new['prototype_capacity']['prepared_cache_sha256']='changed'
    elif change=='width':new['prototype_capacity']['specification']['embedding_dim']=256
    elif change=='aux_weight':new['prototype_capacity']['specification']['target_alignment_weight']=.2
    elif change=='patience':new['baseline']['policy']['patience_epochs']=10
    else:new['model']['dim_proj_adv']=512
    with pytest.raises(ValueError):assert_declared_changes(new,old,'prototype')


def report(method):
    p=dict(model_updated=True,target_gt_losses=[0.,0.,0.],source_gradient_norm=.5)
    if method=='prototype':p.update(branch_parameters_updated=True,auxiliary=dict(rq3_loss=.51,rq3_source_loss=.5,
        rq3_alignment_loss=.1,rq3_target_gradient_norm=.001,rq3_selected=1.))
    return dict(status='passed',batch=2,initial_detector_sha='same',checked_target_batches=16,
        rng_roundtrip={'status':'passed'},cuda_seed_proof={'actual_cuda_seed':123},phases={'1':deepcopy(p),'5':deepcopy(p)})


@pytest.mark.parametrize('method',['uda','prototype'])
def test_acceptance_uses_native_report_formats_and_rejects_target_gt(method):
    r=report(method);validate_probe(r,method,'same')
    r['phases']['1']['target_gt_losses'][0]=.1
    with pytest.raises(ValueError):validate_probe(r,method,'same')


@pytest.mark.parametrize('change',['initial','seed','rng','branch','gradient','weight'])
def test_acceptance_rejects_mismatched_or_zero_auxiliary_evidence(change):
    r=report('prototype')
    if change=='initial':r['initial_detector_sha']='changed'
    elif change=='seed':r['cuda_seed_proof']['actual_cuda_seed']=6
    elif change=='rng':r['rng_roundtrip']['status']='failed'
    elif change=='branch':r['phases']['5']['branch_parameters_updated']=False
    elif change=='gradient':r['phases']['5']['auxiliary']['rq3_target_gradient_norm']=0.
    else:r['phases']['5']['auxiliary']['rq3_loss']=.52
    with pytest.raises(ValueError):validate_probe(r,'prototype','same')


def test_true_sada_uses_new_level_weights_without_changing_detection_loss(monkeypatch):
    import torch
    from runpy import run_path
    from tools.baseline_protocol import CODE
    from libs.models.SADA.meta_archs import SADA
    small=run_path(str(CODE/'tests/test_corrected_baseline.py'))['small_config']
    native=torch.tensor
    def cpu_copy(*args,**kwargs):
        value=native(*args,**kwargs)
        return value.clone() if value.requires_grad and value.device.type=='cpu' else value
    monkeypatch.setattr(torch,'tensor',cpu_copy)
    old=small();new=deepcopy(old)
    for i,w in enumerate(LEVELS):new['model']['weight_level_%d'%i]=w
    new['model']['pl_threshold']=THRESHOLD
    torch.manual_seed(123);a=SADA(old,'cpu')
    torch.manual_seed(123);b=SADA(new,'cpu')
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    def video(name,labelled):
        return dict(feats=torch.randn(8,64),segments=torch.tensor([[10.,35.]]) if labelled else torch.empty(0,2),
            labels=torch.tensor([0]) if labelled else torch.empty(0,dtype=torch.long),
            metadata=dict(video_id=name,feat_stride=16,feat_num_frames=32,fps=30,duration=60))
    source,target=[video('s',True)],[video('t',False)];out=[]
    for model in (a,b):
        torch.manual_seed(777);out.append(model(video_list_src=deepcopy(source),video_list_tgt=deepcopy(target),mode='train',curr_epoch=5)[0])
    assert torch.equal(out[0]['cls_loss'],out[1]['cls_loss']) and torch.equal(out[0]['reg_loss'],out[1]['reg_loss'])
    expected=sum(w*out[1]['loss_conditional_adv_rev_%d'%i]/6 for i,w in enumerate(LEVELS))
    torch.testing.assert_close(out[1]['loss_conditional_adv_rev'],expected)
    assert expected>0 and b.cfg['model']['pl_threshold']==.5
    (out[1]['downstream_loss']+out[1]['loss_conditional_adv_rev']).backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in b.parameters())

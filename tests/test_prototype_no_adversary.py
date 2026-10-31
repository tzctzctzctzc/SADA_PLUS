from copy import deepcopy
from pathlib import Path
from runpy import run_path
import subprocess

import pytest
import torch

from tools.run_prototype_no_adversary import changed_config,assert_change,SHIFTS,CONTROL,BASE,read,sha
from tools.baseline_protocol import CODE
from libs.models.SADA.prototype_capacity import specification


def old_config():
    return dict(seed=123,init_rand_seed=123,loader=dict(batch_size=2),
        model=dict(adv_method=['conditional_rev'],loss_weights=dict(weight_adv=2.)),
        baseline=dict(shift='S2',regime='uda',code={},tmpdir='/old',target_manifest='/real-target.json'),
        prototype_capacity=dict(variant='control',specification=specification('control'),
            prepared_cache='/frozen-cache',prepared_cache_sha256='same'),opt=dict(learning_rate=.0001,epochs=50))


def test_only_disable_adversary_and_keep_real_target_stream():
    old=old_config();before=deepcopy(old);cfg=changed_config(old,{'commit':'new'},'/new')
    assert_change(cfg,old)
    assert cfg['model']['adv_method']==['None']
    assert cfg['baseline']['regime']=='uda' and cfg['baseline']['target_manifest']=='/real-target.json'
    assert cfg['model']['loss_weights']==old['model']['loss_weights'] and old==before


@pytest.mark.parametrize('change',['regime','target','seed','batch','lr','width','cache','weight','threshold','budget'])
def test_extra_changes_rejected(change):
    old=old_config();cfg=changed_config(old,{},'/new')
    if change=='regime':cfg['baseline']['regime']='so'
    elif change=='target':cfg['baseline']['target_manifest']='/source-in-disguise'
    elif change=='seed':cfg['seed']=6
    elif change=='batch':cfg['loader']['batch_size']=4
    elif change=='lr':cfg['opt']['learning_rate']=.0002
    elif change=='width':cfg['prototype_capacity']['specification']['embedding_dim']=256
    elif change=='cache':cfg['prototype_capacity']['prepared_cache_sha256']='changed'
    elif change=='weight':cfg['prototype_capacity']['specification']['target_alignment_weight']=.2
    elif change=='threshold':cfg['model']['pl_threshold']=.1
    else:cfg['opt']['epochs']=25
    with pytest.raises(ValueError):assert_change(cfg,old)


def test_no_model_data_trainer_or_engine_changes():
    changed=subprocess.check_output(['git','diff','23d553e26b80f2263bc019caa5b9100020732903',
        '--name-only','--','libs/','configs/','tools/baseline_engine.py','tools/run_sada_baseline.py'],cwd=CODE,text=True)
    assert not changed.strip()


def test_true_sada_preserves_losses_and_initialization_but_has_no_discriminator_gradients(tmp_path,monkeypatch):
    from libs.models.SADA.meta_archs import SADA
    from tools.baseline_engine import tensor_digest
    from libs.models.SADA.optimizer import make_optimizer
    helpers=run_path(str(CODE/'tests/test_prototype_capacity.py'))
    small=run_path(str(CODE/'tests/test_corrected_baseline.py'))['small_config']
    native=torch.tensor
    def cpu_copy(*args,**kwargs):
        value=native(*args,**kwargs)
        return value.clone() if value.requires_grad and value.device.type=='cpu' else value
    monkeypatch.setattr(torch,'tensor',cpu_copy)
    cfg=small();cfg['prototype_capacity']=helpers['configuration'](tmp_path,'control')
    plain=deepcopy(cfg);plain['model']['adv_method']=['None']
    torch.manual_seed(123);original=SADA(cfg,'cpu')
    torch.manual_seed(123);model=SADA(plain,'cpu')
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in original.state_dict().items())
    source,target=helpers['videos']();target['feats'].requires_grad_()
    torch.manual_seed(777);a=original(video_list_src=[source],video_list_tgt=[target],mode='train',curr_epoch=5)[0]
    torch.manual_seed(777);b=model(video_list_src=[source],video_list_tgt=[target],mode='train',curr_epoch=5)[0]
    for key in ('cls_loss','reg_loss','rq3_loss','rq3_source_loss','rq3_alignment_loss'):
        torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
    assert 'loss_conditional_adv_rev' not in b and b['rq3_selected']>0 and b['rq3_target_gradient_norm']>0
    disc_names=('disc_dom_adv','disc_dom_conditional_adv','learnable_embeddings')
    disc_before=tensor_digest({k:v for k,v in model.state_dict().items() if any(n in k for n in disc_names)})
    optimizer=make_optimizer(model,cfg['opt'],.0001,exclude_params=list(disc_names))
    disc_optimizer=make_optimizer(model,cfg['opt'],.0001,include_params=list(disc_names))
    # 保留原适配阶段对downstream_loss的总系数，不顺便改变源监督强度。
    loss=plain['model']['loss_weights']['weight_downstream']*b['downstream_loss'];loss.backward()
    assert model.rq3_capacity.branch.input_projection.weight.grad.abs().sum()>0
    assert target['feats'].grad.abs().sum()>0
    assert all(p.grad is None for n,p in model.named_parameters() if any(k in n for k in disc_names))
    optimizer.step();disc_optimizer.step()
    assert not disc_optimizer.state_dict()['state']
    assert disc_before==tensor_digest({k:v for k,v in model.state_dict().items() if any(n in k for n in disc_names)})


@pytest.mark.skipif(not CONTROL.exists(),reason='需要远端真实冻结六方向工件')
def test_all_six_real_configs_and_teacher_cache_unchanged():
    for shift in SHIFTS:
        old=read(CONTROL/shift/'config.json');new=changed_config(old,{'commit':'test'},'/tmp')
        assert_change(new,old)
        target=read(new['baseline']['target_manifest'])['database']
        assert target and all('annotations' not in row for row in target.values())
        assert new['baseline']['target_manifest']==read(BASE/shift/'uda/config.json')['baseline']['target_manifest']
        cap=new['prototype_capacity'];assert sha(cap['prepared_cache'])==cap['prepared_cache_sha256']

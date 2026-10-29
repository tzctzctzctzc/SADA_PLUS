"""真实错误复现、官方数值等价与早停/隔离边界；不以target指标调参。"""
import ast
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import types

import numpy as np
import pytest
import torch
import yaml

from tools.baseline_protocol import (CODE,OFFICIAL,assert_no_target_gt,batch_choice,
                                      decision,sanitized_inputs,termination,save,read,audit_pair)
from tools.run_sada_baseline import budget,runtime_config,membership_manifest


def test_earlystop_means_epochs_not_validation_counts():
    history=[[5,1.],[10,.8],[15,.81],[20,.82],[25,.83]]
    assert termination(history)==dict(epoch=25,best_epoch=10,best_loss=.8,reason='source_patience_exhausted')


def test_no_premature_or_late_terminal():
    with pytest.raises(ValueError):termination([[5,1.],[10,.8],[15,.9]])
    with pytest.raises(ValueError):termination([[5,1.],[10,1.],[15,1.],[20,1.],[25,.5]])
    with pytest.raises(ValueError):decision([[10,1.]])


def test_ties_do_not_reset_patience():
    assert termination([[5,1.],[10,1.],[15,1.],[20,1.]])['epoch']==20


def test_isolation_and_membership():
    db={'s':dict(subset='s',annotations=[{'label':'take','label_id':0,'segment':[1,2]}]),
        'v':dict(subset='v',annotations=[]),'t':dict(subset='t',fps=30,duration=100,
            annotations=[{'label':'secret','label_id':9,'segment':[99,100]}])}
    source,target=sanitized_inputs(db,['s'],['v'],['t'])
    assert set(source['database'])=={'s','v'}
    assert set(target['database']['t'])=={'subset','fps','duration'}
    db['t']['annotations']=[{'label':'changed','label_id':6,'segment':[1,99]}]
    assert (source,target)==sanitized_inputs(db,['s'],['v'],['t'])


def test_gt_rejected_before_model():
    assert_no_target_gt([{'segments':torch.empty(0,2),'labels':torch.empty(0,dtype=torch.long)}])
    with pytest.raises(ValueError):assert_no_target_gt([{'segments':torch.ones(1,2),'labels':torch.zeros(1)}])


def test_manifest_preserves_video_order(tmp_path):
    path=tmp_path/'source.json'
    save(path,{'database':{'z':{'subset':'source'},'a':{'subset':'source'}}},sort_keys=False)
    assert list(read(path)['database'])==['z','a']


def test_membership_ledger_uses_video_ids_not_json_wrapper_keys():
    db={name:{'subset':split} for name,split in [('s','s_train'),('sv','s_val'),
        ('target_video_a','t_train'),('target_video_b','t_train'),('tv','t_val')]}
    cfg=dict(src_train_split=['s_train'],src_val_split=['s_val'],
             tgt_train_split=['t_train'],tgt_val_split=['t_val'])
    result=membership_manifest(db,cfg)
    assert result['target_train']==['target_video_a','target_video_b']
    assert result['source_train']==['s'] and result['target_val']==['tv']


def test_pair_rejects_batch_or_initialization_changes(tmp_path):
    roots=[tmp_path/x for x in ['so','uda']]
    for root,regime in zip(roots,['so','uda']):
        save(root/'config.json',dict(loader={'batch_size':4},baseline={'regime':regime,'shift':'S1'}))
        save(root/'termination.json',dict(initial_state_sha='same'))
    assert audit_pair(*roots)['status']=='passed'
    changed=read(roots[1]/'config.json');changed['loader']['batch_size']=2
    save(roots[1]/'config.json',changed)
    with pytest.raises(ValueError):audit_pair(*roots)
    changed['loader']['batch_size']=4;save(roots[1]/'config.json',changed)
    save(roots[1]/'termination.json',dict(initial_state_sha='different'))
    with pytest.raises(ValueError):audit_pair(*roots)


def test_batch_frozen_by_both_phases():
    reports={2:{'phases':{'1':{'samples_per_second':10},'5':{'samples_per_second':8}}},
             4:{'phases':{'1':{'samples_per_second':20},'5':{'samples_per_second':7}}}}
    assert batch_choice(reports)==2
    reports[4]['phases']['5']['samples_per_second']=12
    assert batch_choice(reports)==4


def test_matched_update_budget_and_lr_not_scaled():
    db={str(i):{'subset':'s' if i<12 else 't'} for i in range(32)}
    cfg={'src_train_split':['s'],'tgt_train_split':['t']}
    assert budget(db,cfg,4)['iterations_per_epoch']==5
    from tools.run_sada_baseline import runtime_config
    cfg.update(loader={'batch_size':2},opt={'learning_rate':.0001},
               baseline={'batch_budget_candidates':{'4':budget(db,cfg,4)}})
    so,uda=[runtime_config(cfg,4,r) for r in ['so','uda']]
    assert so['baseline']['epoch_budget']==uda['baseline']['epoch_budget']
    assert so['opt']['learning_rate']==.0001 and cfg['loader']['batch_size']==2


def dataset_fixture(tmp_path,training=False,unlabeled=False):
    from libs.datasets.epic_kitchens import EpicKitchensDataset
    np.savez(tmp_path/'s.npz',feats=np.ones((128,4),dtype=np.float32))
    payload={'database':{'s':dict(subset='train',fps=1,duration=128,
        annotations=[{'label':'a','label_id':0,'segment':[1,10]},
                     {'label':'b','label_id':1,'segment':[3,4]}])}}
    (tmp_path/'db.json').write_text(json.dumps(payload))
    return EpicKitchensDataset(training,['train'],str(tmp_path),str(tmp_path/'db.json'),
        1,2,1,1,64,.3,None,4,10,None,'.npz',False,
        action_subsampling='10_majority_classes',unlabeled=unlabeled)


def test_nested_intervals_and_labels_stay_paired(tmp_path):
    video=dataset_fixture(tmp_path)[0]
    assert video['segments'].tolist()==[[0.,9.],[2.,3.]]
    assert video['labels'].tolist()==[0,1]


def test_target_crop_and_frame_units(tmp_path):
    video=dataset_fixture(tmp_path,training=True,unlabeled=True)[0]
    assert video['segments'].numel()==0 and video['labels'].numel()==0
    assert video['feats'].shape[-1]==64
    assert video['metadata']['feat_num_frames']==2
    assert video['metadata']['crop_feat_len']==64


def test_full_split_evaluation(tmp_path):
    from libs.evaluation.evaluator_anet import load_gt_seg_from_json
    db={'database':{str(i):dict(subset=s,annotations=[dict(label_id=0,segment=[1,2])])
                    for i,s in enumerate(['one','two','other'])}}
    path=tmp_path/'db.json';path.write_text(json.dumps(db))
    result=load_gt_seg_from_json(path,['one','two'])
    assert set(result['video-id'])=={'0','1'}


def test_device_transfer_does_not_mutate_cached_batch():
    # 执行实际函数AST，避免这个纯搬运测试依赖wandb等日志包。
    tree=ast.parse((CODE/'libs/trainer/da_trainer.py').read_text(encoding='utf-8'))
    node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='move_to_device')
    namespace={'torch':torch}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'move_to_device.py','exec'),namespace)
    move_to_device=namespace['move_to_device']
    source=[{'feats':torch.ones(4,8),'metadata':{'video_id':'s'}}]
    result=move_to_device(source,'cpu')
    assert result is not source and result[0] is not source[0]
    assert torch.equal(result[0]['feats'],source[0]['feats'])


def test_official_epoch_and_optimizer_unchanged():
    before=subprocess.check_output(['git','show',OFFICIAL+':libs/trainer/da_trainer.py'],cwd=CODE,text=True)
    after=(CODE/'libs/trainer/da_trainer.py').read_text(encoding='utf-8')
    def method(text):
        cls=next(x for x in ast.parse(text).body if isinstance(x,ast.ClassDef) and x.name=='Trainer')
        return ast.dump(next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_train_epoch'))
    assert method(before)==method(after)
    for path in ['libs/models/SADA/optimizer.py','libs/models/SADA/backbones.py',
                 'libs/models/SADA/blocks.py','libs/models/SADA/losses_aux.py']:
        assert subprocess.check_output(['git','show',OFFICIAL+':'+path],cwd=CODE)==(CODE/path).read_bytes().replace(b'\r\n',b'\n')


def small_config():
    cfg=yaml.safe_load((CODE/'configs/dark_vs_all_new_videos_full/version_final_ckpt.yaml').read_text())
    cfg['dataset'].update(max_seq_len=64,input_dim=8)
    cfg['model'].update(embd_dim=32,fpn_dim=32,head_dim=32,sgp_mlp_dim=64,dim_proj_adv=16,
        input_dim=8,num_classes=97,test_cfg=cfg['test_cfg'],max_seq_len=64)
    return cfg


def test_true_sada_loss_gradient_identical_to_official(monkeypatch):
    from libs.models.SADA.meta_archs import SADA
    original=subprocess.check_output(['git','show',OFFICIAL+':libs/models/SADA/meta_archs.py'],cwd=CODE,text=True)
    namespace={'__name__':'libs.models.SADA.official_comparison','__package__':'libs.models.SADA'}
    # 独立载入官方类，随后恢复注册表，正式入口始终使用修正版。
    from libs.models.SADA.models import meta_archs
    try:exec(compile(original,'official_meta_archs.py','exec'),namespace)
    finally:meta_archs['sada']=SADA
    # 官方常量.to(CUDA)是非叶子；CPU的.to(CPU)不拷贝，原地累加会报错。
    # 两个版本使用同一identity拷贝模拟此设备行为，不修改正式模型或NMS。
    native_tensor=torch.tensor
    def cpu_copy(*args,**kwargs):
        value=native_tensor(*args,**kwargs)
        return value.clone() if value.requires_grad and value.device.type=='cpu' else value
    monkeypatch.setattr(torch,'tensor',cpu_copy)
    cfg=small_config()
    torch.manual_seed(6);a=namespace['SADA'](deepcopy(cfg),'cpu')
    torch.manual_seed(6);b=SADA(deepcopy(cfg),'cpu')
    assert set(a.state_dict())==set(b.state_dict())
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    def video(name,labelled):
        return dict(feats=torch.randn(8,64),segments=torch.tensor([[10.,35.]]) if labelled else torch.empty(0,2),
                    labels=torch.tensor([0]) if labelled else torch.empty(0,dtype=torch.long),
                    metadata={'video_id':name,'feat_stride':16,'feat_num_frames':32,'fps':30,'duration':60})
    source,target=[video('s',True)],[video('t',False)]
    for epoch in [1,5]:
        outputs=[];grads=[]
        for model in [a,b]:
            model.train();model.zero_grad(set_to_none=True);torch.manual_seed(100+epoch)
            losses=model(video_list_src=deepcopy(source),video_list_tgt=deepcopy(target),mode='train',curr_epoch=epoch)[0]
            value=losses['downstream_loss']*(1 if epoch<5 else 2)
            if epoch>=5:value=value+losses['loss_conditional_adv_rev']
            value.backward();outputs.append(losses)
            grads.append({k:p.grad.clone() for k,p in model.named_parameters() if p.grad is not None})
        assert outputs[0].keys()==outputs[1].keys()
        assert all(torch.equal(outputs[0][k],outputs[1][k]) for k in outputs[0])
        assert grads[0].keys()==grads[1].keys()
        # 共享参数的反传累加顺序可有末位差异；不将数值等价伪称位级一致。
        for k in grads[0]:
            assert torch.isfinite(grads[0][k]).all() and torch.isfinite(grads[1][k]).all()
            torch.testing.assert_close(grads[0][k],grads[1][k],rtol=1e-6,atol=1e-8)

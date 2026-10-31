"""检查真实数据接口语义和SO/UDA配对，不以目标结果调整配置。"""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch

from libs.datasets.charades_features import CharadesFeatures
from tools.run_charades_baselines import template,BENCHMARK
from tools.run_sada_baseline import runtime_config


def loader(tmp_path,unlabeled,training=False):
    np.savez(tmp_path/'v.npz',feats=np.ones((1 if not training else 50,1024),dtype=np.float32))
    path=tmp_path/'db.json'
    path.write_text(json.dumps({'database':{'v':dict(subset='train',fps=24,duration=16,
        annotations=[dict(label='SIT',label_id=2,segment=[1.,3.])])}}))
    return CharadesFeatures(training,['train'],feat_folder=str(tmp_path),json_file=str(path),
        feat_stride=8,num_frames=32,default_fps=24,downsample_rate=1,max_seq_len=16,
        trunc_thresh=.3,crop_ratio=[.9,1.],input_dim=1024,num_classes=10,
        file_prefix='',file_ext='.npz',force_upsampling=False,action_subsampling='all',unlabeled=unlabeled)


def test_target_training_ignores_annotations_and_retains_frame_units(tmp_path):
    ds=loader(tmp_path,True,True);v=ds[0]
    assert v['segments'].shape==(0,2) and v['labels'].shape==(0,)
    assert v['feats'].shape==(1024,16)
    assert v['metadata']['feat_num_frames']==32 and v['metadata']['crop_feat_len']==16
    assert ds.get_attributes()['dataset_name']==BENCHMARK


def test_source_seconds_to_grid_and_single_timestep(tmp_path):
    ds=loader(tmp_path,False);v=ds[0]
    assert v['feats'].shape==(1024,1)
    assert v['segments'].tolist()==[[1.,7.]] and v['labels'].tolist()==[2]


def test_background_source_has_empty_tensor_instead_of_none(tmp_path):
    ds=loader(tmp_path,False)
    ds.data_list[0]['segments']=np.empty((0,2),dtype=np.float32)
    ds.data_list[0]['labels']=np.empty(0,dtype=np.int64)
    v=ds[0]
    assert v['segments'].shape==(0,2) and v['labels'].dtype==torch.int64


def test_official_model_and_paired_budget(tmp_path):
    import hashlib
    from tools.baseline_protocol import POLICY
    for name in ('source.json','target_unlabeled.json'):(tmp_path/name).write_text('{}')
    splits=['third_training']*5+['third_validation']+['ego_training']*6+['ego_validation']
    db={str(i):{'subset':s} for i,s in enumerate(splits)}
    (tmp_path/'offline_evaluation.json').write_text(json.dumps({'database':db}))
    (tmp_path/'identity.json').write_text(json.dumps(dict(evaluation_protocol='clean_only',interval_policy='clean')))
    (tmp_path/'inventory.json').write_text('{}')
    cfg=template(tmp_path,tmp_path,{'commit':'test'},tmp_path/'tmp')
    so,uda=[runtime_config(cfg,2,r) for r in ('so','uda')]
    assert so['benchmark']==BENCHMARK and so['seed']==123 and so['loader']['batch_size']==2
    assert so['model']['embd_dim']==256 and so['dataset']['input_dim']==1024
    assert so['train_cfg']['warmup_adv']==0 and so['baseline']['policy']==POLICY
    assert so['baseline']['epoch_budget']['iterations_per_epoch']==3
    for value in (so,uda):value['baseline'].pop('regime')
    assert so==uda

"""Charades-Ego第三→第一：先独立验收，再配对训练SO/UDA；不等待原视频。"""
import argparse
from copy import deepcopy
import math
import os
from pathlib import Path
import signal
import sys
import time

CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE))
from tools import run_sada_baseline as base
from tools.prepare_charades_features import relocated_file_name
from tools.baseline_protocol import (POLICY,CUDA_RNG_POLICY,random_seeds,code_identity,
    read,save,sha,audit_entry,audit_pair,termination)

SCHEMA='charades-ego-corrected-source-earlystop-v1'
BENCHMARK='charades-ego-third-to-first'
SHIFT='third_to_first'


def template(inputs,features,identity,tmpdir):
    import yaml
    cfg=yaml.safe_load((CODE/'configs/charadesego/version_final.yaml').read_text())
    original=deepcopy(cfg)
    cfg.update(seed=123,init_rand_seed=123,gpu=0,benchmark=BENCHMARK)
    cfg['dataset']=dict(feat_folder=str(features),json_file=str(inputs/'source.json'),
        feat_stride=8,num_frames=32,default_fps=24,downsample_rate=1,max_seq_len=2304,
        trunc_thresh=.3,crop_ratio=[.9,1.],input_dim=1024,num_classes=10,file_prefix='',
        file_ext='.npz',force_upsampling=False,action_subsampling='all')
    cfg['model'].update(input_dim=1024,num_classes=10,test_cfg=cfg['test_cfg'],max_seq_len=2304)
    # 发布Charades配置没有warmup_adv；其训练/模型均以.get(...,0)执行。
    cfg['train_cfg'].update(valid_freq=5,test_freq=-1,early_stop=-1,warmup_adv=0)
    cfg['loader']['batch_size']=2
    metadata=read(inputs/'identity.json')
    if metadata['evaluation_protocol']!='clean_only' or metadata['interval_policy']!='clean':
        raise ValueError('必须使用已确认的清洗后协议')
    cfg['baseline']=dict(schema=SCHEMA,shift=SHIFT,policy=POLICY,code=identity,
        random_seeds=random_seeds(123),cuda_rng_policy=CUDA_RNG_POLICY,tmpdir=str(tmpdir),
        official_config=str(CODE/'configs/charadesego/version_final.yaml'),
        official_config_sha256=sha(CODE/'configs/charadesego/version_final.yaml'),
        annotation_for_offline_evaluation=str(inputs/'offline_evaluation.json'),
        annotation_sha256=sha(inputs/'offline_evaluation.json'),
        source_manifest_sha256=sha(inputs/'source.json'),target_manifest=str(inputs/'target_unlabeled.json'),
        target_manifest_sha256=sha(inputs/'target_unlabeled.json'),target_gt_used=False,
        source_target_budget='max-source-target-batches',initialization='official-independent-EMA',
        annotation_policy='clean_only_no_additional_deduplication',
        dataset_identity_sha256=sha(inputs/'identity.json'),
        feature_inventory=str(features/'inventory.json'),feature_inventory_sha256=sha(features/'inventory.json'))
    db=read(inputs/'offline_evaluation.json')['database']
    cfg['baseline']['batch_budget_candidates']={'2':base.budget(db,cfg,2)}
    # 仅允许上述接口/预算管理改动，模型结构、优化器和学习率保持作者Charades配置。
    expected=deepcopy(original['model'])
    expected.update(input_dim=1024,num_classes=10,test_cfg=cfg['test_cfg'],max_seq_len=2304)
    if cfg['model']!=expected or cfg['opt']!=original['opt']:raise ValueError('作者模型/优化器配置改变')
    return cfg


def input_audit(root,inputs,features,identity,tmpdir):
    from tools.baseline_engine import dataset
    import numpy as np
    import torch
    cfg=template(inputs,features,identity,tmpdir)
    metadata=read(inputs/'identity.json')
    for name,digest in metadata['files'].items():
        if sha(inputs/relocated_file_name(name))!=digest:raise ValueError('标注工件SHA变化')
    db=read(inputs/'offline_evaluation.json')['database']
    members=base.membership_manifest(db,cfg)
    if any(set(a)&set(b) for i,a in enumerate(members.values()) for b in list(members.values())[i+1:]):
        raise ValueError('视频成员交叉')
    target=read(inputs/'target_unlabeled.json')['database']
    if set(target)!=set(members['target_train']) or any('annotations' in v for v in target.values()):
        raise ValueError('目标训练清单不是无标注精确成员')
    inventory=read(features/'inventory.json')
    if {r['video_id'] for r in inventory['files']}!=set(db):raise ValueError('特征不覆盖精确成员')
    for row in inventory['files']:
        if sha(features/(row['video_id']+'.npz'))!=row['sha256']:raise ValueError('迁移特征SHA不符')
    checks=[]
    for role,splits,path,unlabeled in [
        ('source_train',cfg['src_train_split'],inputs/'source.json',False),
        ('source_val',cfg['src_val_split'],inputs/'source.json',False),
        ('target_train',cfg['tgt_train_split'],inputs/'target_unlabeled.json',True)]:
        ds=dataset(cfg,path,splits,False,unlabeled)
        if len(ds)!=len(members[role]):raise ValueError('真实loader成员数不符')
        ids=sorted({0,len(ds)//2,len(ds)-1}|{i for i,r in enumerate(ds.data_list) if r['id']=='HWFK4'})
        for i in ids:
            v=ds[i];vid=v['metadata']['video_id'];annotations=[] if unlabeled else db[vid]['annotations']
            expected=np.asarray([a['segment'] for a in annotations],dtype=np.float32).reshape(-1,2)
            expected=(expected*24-16)/8
            if v['feats'].shape[0]!=1024 or not torch.isfinite(v['feats']).all():raise ValueError('真实特征格式错误')
            if not np.array_equal(v['segments'].numpy(),expected):raise ValueError('真实标注时间坐标错误')
            if v['labels'].tolist()!=[a['label_id'] for a in annotations]:raise ValueError('真实类别对应错误')
            checks.append(dict(role=role,video_id=vid,shape=list(v['feats'].shape),annotations=len(annotations)))
    save(root/'inputs/membership.json',members)
    save(root/'inputs/template.json',cfg)
    save(root/'inputs/identity.json',dict(status='passed',schema=SCHEMA,benchmark=BENCHMARK,
        code=identity,counts=metadata['counts'],real_loader_checks=checks,
        annotation_identity_sha256=sha(inputs/'identity.json'),feature_inventory_sha256=sha(features/'inventory.json'),
        checked_feature_count=len(inventory['files']),target_annotations_present=False))
    return cfg


def complete(root,entry,identity,regime):
    artifacts=[entry/x for x in ('config.json','runtime_config.json','seed_proof.json','metrics.jsonl',
        'termination.json','best.pt','latest.pt','train_gpu.json','evaluate_gpu.json',
        'evaluation/source_val.json','evaluation/target_val.json','evaluation/manifest.json')]
    save(entry/'complete.json',dict(schema=SCHEMA,benchmark=BENCHMARK,shift=SHIFT,regime=regime,
        code=identity,acceptance_sha256=sha(root/'acceptance.json'),artifacts={str(p):sha(p) for p in artifacts}))
    return audit_entry(entry,expected_schema=SCHEMA)


def main(args):
    identity=code_identity();root=args.output_root.resolve();data=args.data_root.resolve()
    root.mkdir(parents=True,exist_ok=False);args.tmpdir.mkdir(parents=True,exist_ok=True)
    save(root/'state.json',dict(phase='cpu_input_audit',pid=os.getpid(),completed=[],active={}))
    cfg=input_audit(root,args.inputs.resolve(),args.features.resolve(),identity,args.tmpdir.resolve())
    if args.prepare_only:return
    indices=[int(i) for i in args.gpus.split(',')]
    reports={}
    for regime in ('uda','so'):
        save(root/'state.json',dict(phase='real_data_acceptance',regime=regime,pid=os.getpid(),completed=[],active={}))
        reports[regime]=base.synchronous_probe(root,data,base.runtime_config(cfg,2,regime),indices,16384,regime+'_acceptance')
        report=reports[regime]
        if report['rng_roundtrip']['status']!='passed' or report['cuda_seed_proof']['actual_cuda_seed']!=123:
            raise ValueError('真实GPU播种/完整恢复核验失败')
        if any(p['source_gradient_norm']<=0 or p['target_gt_losses']!=[0.,0.,0.] for p in report['phases'].values()):
            raise ValueError('真实监督/GT隔离核验失败')
    if reports['so']['initial_state_sha']!=reports['uda']['initial_state_sha']:
        raise ValueError('SO/UDA冷初始化不同')
    minimum=max(12288,math.ceil(max(r['peak_reserved_mib'] for r in reports.values())+4096))
    save(root/'acceptance.json',dict(status='passed',batch=2,minimum_free_mib=minimum,code=identity,
        inputs_sha256=sha(root/'inputs/identity.json'),reports=reports,
        report_sha256={str(p):sha(p) for p in (root/'probes').glob('*/result.json')}))
    save(root/'plan.json',dict(schema=SCHEMA,benchmark=BENCHMARK,seed=123,batch=2,policy=POLICY,
        jobs=[(SHIFT,r) for r in ('so','uda')],prototype='wait_for_raw_video_and_new_teacher_cache'))
    pending=['so','uda'];completed=[]
    try:
        while pending or base.ACTIVE:
            for regime,task in list(base.ACTIVE.items()):
                code=task['process'].poll()
                if code is None:continue
                if code:raise RuntimeError('阶段失败: '+regime+'/'+task['stage'])
                entry=task['entry']
                if task['stage']=='train':
                    termination(read(entry/'termination.json')['history'])
                    task['stage']='evaluate';task['process']=base.spawn('evaluate',entry/'config.json',entry,task['lease'],data)
                else:
                    completed.append(complete(root,entry,identity,regime))
                    task['context'].__exit__(None,None,None);del base.ACTIVE[regime]
                    print('COMPLETE',regime,completed[-1]['map'],flush=True)
            while pending:
                found=base.acquire(indices,data,root,minimum,identity['commit'])
                if found is None:break
                context,lease=found;regime=pending.pop(0);entry=root/'formal'/SHIFT/regime
                entry.mkdir(parents=True,exist_ok=False);save(entry/'config.json',base.runtime_config(cfg,2,regime))
                base.ACTIVE[regime]=dict(context=context,lease=lease,entry=entry,stage='train',
                    process=base.spawn('train',entry/'config.json',entry,lease,data))
            save(root/'state.json',dict(phase='formal',schema=SCHEMA,pid=os.getpid(),completed=completed,pending=pending,
                active={k:dict(pid=t['process'].pid,gpu=t['lease']['index'],stage=t['stage']) for k,t in base.ACTIVE.items()}))
            time.sleep(5)
    except BaseException:
        for task in base.ACTIVE.values():
            if task['process'].poll() is None:os.killpg(task['process'].pid,signal.SIGTERM)
        raise
    save(root/'pair_audit.json',audit_pair(root/'formal'/SHIFT/'so',root/'formal'/SHIFT/'uda'))
    save(root/'final_audit.json',dict(status='passed',scope='charades-so-uda-only',count=2,results=completed))
    save(root/'state.json',dict(phase='complete_so_uda',completed=completed,active={},prototype_pending=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--features',type=Path,required=True);p.add_argument('--data-root',type=Path,default=Path('/vepfs/vlms/SADA_PLUS'))
    p.add_argument('--tmpdir',type=Path,default=Path('/vepfs/vlms/tmp/charades_1031'));p.add_argument('--gpus',default='0,1,2,3')
    p.add_argument('--prepare-only',action='store_true');args=p.parse_args()
    signal.signal(signal.SIGTERM,base.stop);signal.signal(signal.SIGINT,base.stop)
    try:main(args)
    except BaseException as error:
        if args.output_root.exists():save(args.output_root/'error.json',dict(type=type(error).__name__,error=str(error)))
        raise
